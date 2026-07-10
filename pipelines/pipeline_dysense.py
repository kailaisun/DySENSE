import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from PIL import Image
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.utils.outputs import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer
from utils.utils import encode_seg

from .modeling_uvit import DySENSEModel

logger = logging.get_logger(__name__)


@dataclass
class ImageLabelPipelineOutput(BaseOutput):

    images: Optional[Union[List[Image.Image], np.ndarray]]
    labels: Optional[Union[List[Image.Image], np.ndarray]]
    vae_trajectory: Optional[List[torch.Tensor]] = None
    clip_trajectory: Optional[List[torch.Tensor]] = None


class DySENSEPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "image_encoder->unet->image_vae->label_vae"

    def __init__(
        self,
        image_vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        clip_tokenizer: CLIPTokenizer,
        image_encoder: CLIPVisionModelWithProjection,
        clip_image_processor: CLIPImageProcessor,
        label_vae: AutoencoderKL,
        unet: DySENSEModel,
        scheduler: KarrasDiffusionSchedulers,
        climate_text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            image_vae=image_vae,
            text_encoder=text_encoder,
            clip_tokenizer=clip_tokenizer,
            image_encoder=image_encoder,
            clip_image_processor=clip_image_processor,
            label_vae=label_vae,
            unet=unet,
            scheduler=scheduler,
        )

        self.climate_text_encoder = climate_text_encoder

        self.vae_scale_factor = 2 ** (len(self.image_vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.num_channels_latents = image_vae.config.latent_channels  # noqa
        self.text_encoder_seq_len = getattr(unet, "num_text_tokens", text_encoder.config.max_position_embeddings)
        self.text_encoder_hidden_size = text_encoder.config.hidden_size
        self.image_encoder_projection_dim = image_encoder.config.projection_dim
        self.unet_resolution = unet.sample_size

    def prepare_extra_step_kwargs(self, generator, eta):

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def enable_vae_slicing(self):

        self.image_vae.enable_slicing()
        self.label_vae.enable_slicing()

    def disable_vae_slicing(self):

        self.image_vae.disable_slicing()
        self.label_vae.disable_slicing()

    def enable_vae_tiling(self):

        self.image_vae.enable_tiling()
        self.label_vae.enable_tiling()

    def disable_vae_tiling(self):

        self.image_vae.disable_tiling()
        self.label_vae.disable_tiling()

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):

        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.clip_tokenizer)

            text_inputs = self.clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.clip_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.clip_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.clip_tokenizer.batch_decode(
                    untruncated_ids[:, self.clip_tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.clip_tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.clip_tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.clip_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def encode_climate_prompt(self, climate_prompt, device, num_images_per_prompt, dtype=None):

        if self.climate_text_encoder is None or climate_prompt is None:
            return None

        if isinstance(climate_prompt, str):
            climate_prompt = [climate_prompt]

        with torch.no_grad():
            climate_embeds = self.climate_text_encoder(climate_prompt)

        if dtype is None:
            dtype = self.unet.dtype
        climate_embeds = climate_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = climate_embeds.shape
        climate_embeds = climate_embeds.repeat(1, num_images_per_prompt, 1)
        climate_embeds = climate_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        return climate_embeds

    def prepare_image_vae_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.repeat(batch_size, 1, 1, 1)
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_clip_embeds(
        self, batch_size, clip_img_dim, dtype, device, generator, latents=None
    ):
        shape = (batch_size, 1, clip_img_dim)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.repeat(batch_size, 1, 1)
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_label_latents(
        self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.repeat(batch_size, 1, 1, 1)
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _split(self, x, height, width):

        batch_size = x.shape[0]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        img_vae_dim = self.num_channels_latents * latent_height * latent_width
        label_dim = self.num_channels_latents * latent_height * latent_width

        img_vae, img_clip, label_clip = x.split([img_vae_dim, self.image_encoder_projection_dim, label_dim], dim=1)

        img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents, latent_height, latent_width))
        img_clip = torch.reshape(img_clip, (batch_size, 1, self.image_encoder_projection_dim))
        label = torch.reshape(label_clip, (batch_size, self.num_channels_latents, latent_height, latent_width))
        return img_vae, img_clip, label

    @staticmethod
    def _combine(img_vae, img_clip, label):
        img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))
        img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))
        label = torch.reshape(label, (label.shape[0], -1))
        return torch.concat([img_vae, img_clip, label], dim=-1)

    def _split_joint(self, x, height, width):

        batch_size = x.shape[0]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        text_dim = self.text_encoder_seq_len * self.text_encoder_hidden_size
        img_vae_dim = self.num_channels_latents * latent_height * latent_width
        label_dim = self.num_channels_latents * latent_height * latent_width

        text, img_vae, img_clip, label_clip = x.split([text_dim, img_vae_dim, self.image_encoder_projection_dim, label_dim], dim=1)

        text = torch.reshape(text, (batch_size, self.text_encoder_seq_len, self.text_encoder_hidden_size))
        img_vae = torch.reshape(img_vae, (batch_size, self.num_channels_latents, latent_height, latent_width))
        img_clip = torch.reshape(img_clip, (batch_size, 1, self.image_encoder_projection_dim))
        label = torch.reshape(label_clip, (batch_size, self.num_channels_latents, latent_height, latent_width))
        return text, img_vae, img_clip, label

    @staticmethod
    def _combine_joint(text, img_vae, img_clip, label):

        text = torch.reshape(text, (text.shape[0], -1))
        img_vae = torch.reshape(img_vae, (img_vae.shape[0], -1))
        img_clip = torch.reshape(img_clip, (img_clip.shape[0], -1))
        label = torch.reshape(label, (label.shape[0], -1))
        return torch.concat([text, img_vae, img_clip, label], dim=-1)

    def _unet_forward(self, text, img_vae, img_clip, label, **kwargs):
        return self.unet(text, img_vae, img_clip, label, **kwargs)

    def _get_noise_pred(
        self,
        mode,
        latents,
        t,
        text,
        max_timestep,
        data_type,
        guidance_scale,
        generator,
        device,
        height,
        width,
    ):

        if mode == "text2img":
            img_vae_latents, img_clip_embeds, label_latents = self._split(latents, height, width)

            _, img_vae_out, img_clip_out, label_out = self._unet_forward(
                text, img_vae_latents, img_clip_embeds, label_latents,
                timestep_text=0, timestep_img=t, data_type=data_type,
            )

            x_out = self._combine(img_vae_out, img_clip_out, label_out)

            if guidance_scale <= 1.0:
                return x_out

            text_t = randn_tensor(text.shape, generator=generator, device=device, dtype=text.dtype)

            _, img_vae_out_uncond, img_clip_out_uncond, label_out_uncond = self._unet_forward(
                text_t, img_vae_latents, img_clip_embeds, label_latents,
                timestep_text=max_timestep, timestep_img=t, data_type=data_type,
            )

            x_out_uncond = self._combine(img_vae_out_uncond, img_clip_out_uncond, label_out_uncond)

            return guidance_scale * x_out + (1.0 - guidance_scale) * x_out_uncond
        else:
            text_embeds, image_vae_latents, image_clip_embeds, label_latents = self._split_joint(latents, height, width)

            text_out, img_vae_out, img_clip_out, label_out = self._unet_forward(
                text_embeds, image_vae_latents, image_clip_embeds, label_latents,
                timestep_text=t, timestep_img=t, data_type=data_type,
            )

            x_out = self._combine_joint(text_out, img_vae_out, img_clip_out, label_out)

            if guidance_scale <= 1.0:
                return x_out

            text_t = randn_tensor(text_embeds.shape, generator=generator, device=device, dtype=text_embeds.dtype)
            image_vae_t = randn_tensor(image_vae_latents.shape, generator=generator, device=device, dtype=image_vae_latents.dtype)
            image_clip_t = randn_tensor(image_clip_embeds.shape, generator=generator, device=device, dtype=image_clip_embeds.dtype)
            label_t = randn_tensor(label_latents.shape, generator=generator, device=device, dtype=label_latents.dtype)

            _, img_vae_out_uncond, img_clip_out_uncond, label_out_uncond = self._unet_forward(
                text_t, image_vae_latents, image_clip_embeds, label_latents,
                timestep_text=max_timestep, timestep_img=t, data_type=data_type,
            )

            text_out_uncond, _, _, _ = self._unet_forward(
                text_embeds, image_vae_t, image_clip_t, label_t,
                timestep_text=t, timestep_img=max_timestep, data_type=data_type,
            )

            x_out_uncond = self._combine_joint(text_out_uncond, img_vae_out_uncond, img_clip_out_uncond, label_out_uncond)

            return guidance_scale * x_out + (1.0 - guidance_scale) * x_out_uncond

    @staticmethod
    def check_latents_shape(latents_name, latents, expected_shape):
        latents_shape = latents.shape
        expected_num_dims = len(expected_shape) + 1  # expected dimensions plus the batch dimension
        expected_shape_str = ", ".join(str(dim) for dim in expected_shape)
        if len(latents_shape) != expected_num_dims:
            raise ValueError(
                f"`{latents_name}` should have shape (batch_size, {expected_shape_str}), but the current shape"
                f" {latents_shape} has {len(latents_shape)} dimensions."
            )
        for i in range(1, expected_num_dims):
            if latents_shape[i] != expected_shape[i - 1]:
                raise ValueError(
                    f"`{latents_name}` should have shape (batch_size, {expected_shape_str}), but the current shape"
                    f" {latents_shape} has {latents_shape[i]} != {expected_shape[i - 1]} at dimension {i}."
                )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        latents,
        prompt_embeds,
        vae_latents,
        clip_embeds,
        label_latents,
        negative_prompt_embeds,
    ):
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        if latents is not None:
            individual_latents_available = (
                prompt_embeds is not None or vae_latents is not None or clip_embeds is not None or label_latents is not None
            )
            if individual_latents_available:
                logger.warning(
                    "You have supplied both `latents` and at least one of `prompt_latents`, `vae_latents`, and"
                    " `clip_latents`. The value of `latents` will override the value of any individually supplied latents."
                )
            img_vae_dim = self.num_channels_latents * latent_height * latent_width
            text_dim = self.text_encoder_seq_len * self.text_encoder_hidden_size
            latents_dim = img_vae_dim + self.image_encoder_projection_dim + text_dim
            latents_expected_shape = (latents_dim,)
            self.check_latents_shape("latents", latents, latents_expected_shape)

        if prompt_embeds is not None:
            prompt_embeds_expected_shape = (self.text_encoder_seq_len, self.text_encoder_hidden_size)
            self.check_latents_shape("prompt_embeds", prompt_embeds, prompt_embeds_expected_shape)

        if vae_latents is not None:
            vae_latents_expected_shape = (self.num_channels_latents, latent_height, latent_width)
            self.check_latents_shape("vae_latents", vae_latents, vae_latents_expected_shape)

        if clip_embeds is not None:
            clip_latents_expected_shape = (1, self.image_encoder_projection_dim)
            self.check_latents_shape("clip_embeds", clip_embeds, clip_latents_expected_shape)

        if label_latents is not None:
            label_latents_expected_shape = (self.num_channels_latents, latent_height, latent_width)
            self.check_latents_shape("label_latents", label_latents, label_latents_expected_shape)

        if prompt_embeds is not None and vae_latents is not None and clip_embeds is not None and label_latents is not None:
            if prompt_embeds.shape[0] != vae_latents.shape[0] or prompt_embeds.shape[0] != clip_embeds.shape[0] or prompt_embeds.shape[0] != label_latents.shape[0]:
                raise ValueError(
                    f"All of `prompt_latents`, `vae_latents`, and `clip_latents` are supplied, but their batch"
                    f" dimensions are not equal: {prompt_embeds.shape[0]} != {vae_latents.shape[0]}"
                    f" != {clip_embeds.shape[0]} != {label_latents.shape[0]}."
                )

    @torch.no_grad()
    def __call__(
        self,
        mode: str = "text2img",
        prompt: Union[str, List[str]] = None,
        climate_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        data_type: Optional[int] = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 8.0,
        ignore_label: int = 0,
        use_color_map: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        climate_prompt_embeds: Optional[torch.Tensor] = None,
        vae_latents: Optional[torch.Tensor] = None,
        clip_embeds: Optional[torch.Tensor] = None,
        label_latents: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        replace_vae_latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        replace_clip_latents: Optional[List[torch.Tensor]] = None,
        replace_vae_mode: str = "noised",
        capture_trajectory: bool = False,
    ):
        height = height or self.unet_resolution * self.vae_scale_factor
        width = width or self.unet_resolution * self.vae_scale_factor


        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            latents,
            prompt_embeds,
            vae_latents,
            clip_embeds,
            label_latents,
            negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        if latents is not None:
            prompt_embeds, vae_latents, clip_embeds, label_latents = self._split_joint(latents, height, width)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if climate_prompt_embeds is None:
            climate_prompt_embeds = self.encode_climate_prompt(
                climate_prompt, device, num_images_per_prompt, dtype=prompt_embeds.dtype,
            )
        if climate_prompt_embeds is not None:
            prompt_embeds = torch.cat([prompt_embeds, climate_prompt_embeds], dim=1)

        batch_n = batch_size * num_images_per_prompt
        image_vae_latents = self.prepare_image_vae_latents(
            batch_size=batch_n,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=vae_latents,
        )
        image_clip_embeds = self.prepare_image_clip_embeds(
            batch_size=batch_n,
            clip_img_dim=self.image_encoder_projection_dim,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=clip_embeds,
        )
        label_embeds = self.prepare_label_latents(
            batch_size=batch_n,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=label_latents,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        max_timestep = self.scheduler.config.num_train_timesteps

        if mode == "text2img":
            latents = self._combine(image_vae_latents, image_clip_embeds, label_embeds)
        else:
            assert mode == "joint", f"Invalid mode: {mode}. Choose between 'text2img' and 'joint'."
            latents = self._combine_joint(prompt_embeds, image_vae_latents, image_clip_embeds, label_embeds)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        logger.debug(f"Scheduler extra step kwargs: {extra_step_kwargs}")

        do_replace_vae = replace_vae_latents is not None and mode == "joint"
        replace_is_trajectory = isinstance(replace_vae_latents, list)
        if do_replace_vae and not replace_is_trajectory:
            replace_vae_latents = replace_vae_latents.to(device=device, dtype=image_vae_latents.dtype)
            if replace_vae_latents.shape[0] != batch_n:
                replace_vae_latents = replace_vae_latents.repeat(batch_n, 1, 1, 1)
            replace_fixed_noise = randn_tensor(
                replace_vae_latents.shape, generator=generator, device=device,
                dtype=replace_vae_latents.dtype,
            )

        vae_trajectory = [] if capture_trajectory and mode == "joint" else None
        clip_trajectory = [] if capture_trajectory and mode == "joint" else None

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = self._get_noise_pred(
                    mode,
                    latents,
                    t,
                    prompt_embeds,
                    max_timestep,
                    data_type,
                    guidance_scale,
                    generator,
                    device,
                    height,
                    width,
                )

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if vae_trajectory is not None:
                    _, v_step, c_step, _ = self._split_joint(latents, height, width)
                    vae_trajectory.append(v_step.detach().clone())
                    clip_trajectory.append(c_step.detach().clone())

                if do_replace_vae:
                    text_e, img_vae_e, img_clip_e, label_e = self._split_joint(latents, height, width)
                    if replace_is_trajectory:
                        img_vae_new = replace_vae_latents[i].to(device=device, dtype=img_vae_e.dtype)
                        img_clip_new = (
                            replace_clip_latents[i].to(device=device, dtype=img_clip_e.dtype)
                            if replace_clip_latents is not None else img_clip_e
                        )
                    else:
                        if replace_vae_mode == "clean":
                            img_vae_new = replace_vae_latents
                        elif i + 1 < len(timesteps):
                            next_t = timesteps[i + 1].reshape(1)
                            img_vae_new = self.scheduler.add_noise(
                                replace_vae_latents, replace_fixed_noise, next_t
                            )
                        else:
                            img_vae_new = replace_vae_latents
                        img_clip_new = img_clip_e
                    latents = self._combine_joint(text_e, img_vae_new, img_clip_new, label_e)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if mode == "text2img":
            image_vae_latents, image_clip_embeds, label_latents = self._split(latents, height, width)
        else:
            text_embeds, image_vae_latents, image_clip_embeds, label_latents = self._split_joint(latents, height, width)

        if not output_type == "latent":
            image = self.image_vae.decode(image_vae_latents / self.image_vae.config.scaling_factor, return_dict=False)[0]
            label = self.label_vae.decode(label_latents / self.label_vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = image_vae_latents
            label = label_latents

        self.maybe_free_model_hooks()

        if image is not None:
            do_denormalize = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if label is not None:
            predictions = torch.argmax(label, dim=1)
            probs = torch.nn.functional.softmax(label, dim=1)
            probs = torch.max(probs, dim=1).values
            predictions[probs < 0.5] = ignore_label
            predictions = predictions.cpu().numpy()
            if use_color_map:
                label = encode_seg(predictions).astype(np.uint8)
            else:
                label = predictions.astype(np.uint8)
            label_list = []
            for i in range(label.shape[0]):
                label_list.append(Image.fromarray(label[i]))
            label = label_list

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image, label

        return ImageLabelPipelineOutput(
            images=image, labels=label,
            vae_trajectory=vae_trajectory, clip_trajectory=clip_trajectory,
        )