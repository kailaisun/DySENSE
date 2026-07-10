from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.pipelines.unidiffuser.modeling_uvit import PatchEmbed, trunc_normal_, UTransformer2DModel
from diffusers.utils import logging
from torch import nn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DySENSEModel(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        text_dim: int = 768,
        inner_text_dim: int = 64,
        clip_img_dim: int = 512,
        num_text_tokens: int = 77,
        num_attention_heads: int = 24,
        attention_head_dim: int = 64,
        in_channels: int = 4,
        out_channels: int = 4,
        num_layers: int = 30,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: int = 64,
        num_vector_embeds: Optional[int] = None,
        patch_size: int = 2,
        activation_fn: str = "gelu",
        num_embeds_ada_norm: int = 1000,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        block_type: str = "unidiffuser",
        pre_layer_norm: bool = False,
        use_timestep_embedding: bool = False,
        norm_elementwise_affine: bool = True,
        use_patch_pos_embed: bool = False,
        ff_final_dropout: bool = True,
        use_data_type_embedding: bool = True,
    ):
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim

        assert sample_size is not None, "UniDiffuserModel over patched input must provide sample_size"
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.patch_size = patch_size
        self.num_patches = (self.sample_size // patch_size) * (self.sample_size // patch_size)

        self.pre_text = nn.Linear(text_dim, inner_text_dim)
        self.text_in = nn.Linear(inner_text_dim, self.inner_dim)
        self.vae_img_in = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            use_pos_embed=use_patch_pos_embed,
        )
        self.clip_img_in = nn.Linear(clip_img_dim, self.inner_dim)
        self.vae_label_in = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            use_pos_embed=use_patch_pos_embed,
        )

        self.timestep_img_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_img_embed = (
            TimestepEmbedding(
                self.inner_dim,
                4 * self.inner_dim,
                out_dim=self.inner_dim,
            )
            if use_timestep_embedding
            else nn.Identity()
        )

        self.timestep_text_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_text_embed = (
            TimestepEmbedding(
                self.inner_dim,
                4 * self.inner_dim,
                out_dim=self.inner_dim,
            )
            if use_timestep_embedding
            else nn.Identity()
        )

        self.num_text_tokens = num_text_tokens
        self.num_tokens = 1 + 1 + num_text_tokens + self.num_patches + 1 + self.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.inner_dim))
        self.pos_embed_drop = nn.Dropout(p=dropout)
        trunc_normal_(self.pos_embed, std=0.02)

        self.use_data_type_embedding = use_data_type_embedding
        if self.use_data_type_embedding:
            self.data_type_token_embedding = nn.Embedding(2, self.inner_dim)
            self.data_type_pos_embed_token = nn.Parameter(torch.zeros(1, 1, self.inner_dim))

        self.transformer = UTransformer2DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            block_type=block_type,
            pre_layer_norm=pre_layer_norm,
            norm_elementwise_affine=norm_elementwise_affine,
            use_patch_pos_embed=use_patch_pos_embed,
            ff_final_dropout=ff_final_dropout,
        )

        patch_dim = (patch_size ** 2) * out_channels
        self.text_out = nn.Linear(self.inner_dim, inner_text_dim)
        self.post_text = nn.Linear(inner_text_dim, text_dim)
        self.vae_img_out = nn.Linear(self.inner_dim, patch_dim)
        self.clip_img_out = nn.Linear(self.inner_dim, clip_img_dim)
        self.label_out = nn.Linear(self.inner_dim, patch_dim)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        old_pe = state_dict.get("pos_embed")
        if old_pe is not None and old_pe.shape != self.pos_embed.shape:
            expected = self.pos_embed.shape[1]
            loaded = old_pe.shape[1]
            if loaded < expected:
                diff = expected - loaded
                old_num_text = loaded - (1 + 1 + self.num_patches + 1 + self.num_patches)
                logger.info(
                    f"pos_embed size mismatch ({loaded} vs {expected}). "
                    f"Old num_text_tokens={old_num_text}, new num_text_tokens={self.num_text_tokens}. "
                    f"Zero-filling {diff} new text token positions."
                )
                prefix_len = 1 + 1 + old_num_text
                new_pe = torch.zeros_like(self.pos_embed)
                new_pe[:, :prefix_len, :] = old_pe[:, :prefix_len, :]
                new_pe[:, prefix_len + diff:, :] = old_pe[:, prefix_len:, :]
                state_dict["pos_embed"] = new_pe

        # Drop legacy climate keys that no longer exist in this model
        keys_to_drop = [k for k in state_dict if "climate" in k]
        for k in keys_to_drop:
            del state_dict[k]

        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        image_latents: torch.Tensor,
        image_embeds: torch.Tensor,
        label_latents: torch.Tensor,
        timestep_img: Union[torch.Tensor, float, int],
        timestep_text: Union[torch.Tensor, float, int],
        data_type: Optional[Union[torch.Tensor, float, int]] = 1,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
    ):

        batch_size = image_latents.shape[0]

        text_hidden_states = self.text_in(self.pre_text(prompt_embeds))
        vae_hidden_states = self.vae_img_in(image_latents)
        clip_hidden_states = self.clip_img_in(image_embeds)
        label_hidden_states = self.vae_label_in(label_latents)

        num_text_tokens = text_hidden_states.size(1)
        num_img_tokens = vae_hidden_states.size(1)

        if not torch.is_tensor(timestep_img):
            timestep_img = torch.tensor([timestep_img], dtype=torch.long, device=vae_hidden_states.device)
        timestep_img = timestep_img * torch.ones(batch_size, dtype=timestep_img.dtype, device=timestep_img.device)
        timestep_img_token = self.timestep_img_proj(timestep_img)
        timestep_img_token = timestep_img_token.to(dtype=self.dtype)
        timestep_img_token = self.timestep_img_embed(timestep_img_token)
        timestep_img_token = timestep_img_token.unsqueeze(dim=1)

        if not torch.is_tensor(timestep_text):
            timestep_text = torch.tensor([timestep_text], dtype=torch.long, device=vae_hidden_states.device)
        timestep_text = timestep_text * torch.ones(batch_size, dtype=timestep_text.dtype, device=timestep_text.device)
        timestep_text_token = self.timestep_text_proj(timestep_text)
        timestep_text_token = timestep_text_token.to(dtype=self.dtype)
        timestep_text_token = self.timestep_text_embed(timestep_text_token)
        timestep_text_token = timestep_text_token.unsqueeze(dim=1)

        token_list = [timestep_img_token, timestep_text_token]

        if self.use_data_type_embedding:
            assert data_type is not None
            if not torch.is_tensor(data_type):
                data_type = torch.tensor([data_type], dtype=torch.int, device=vae_hidden_states.device)
            data_type = data_type * torch.ones(batch_size, dtype=data_type.dtype, device=data_type.device)
            data_type_token = self.data_type_token_embedding(data_type).unsqueeze(dim=1)
            token_list.append(data_type_token)

        token_list.extend([text_hidden_states, clip_hidden_states, vae_hidden_states, label_hidden_states])

        hidden_states = torch.cat(token_list, dim=1)

        pe = self.pos_embed
        if self.use_data_type_embedding:
            pe = torch.cat(
                [pe[:, :1 + 1, :], self.data_type_pos_embed_token, pe[:, 1 + 1:, :]],
                dim=1,
            )
        hidden_states = hidden_states + pe
        hidden_states = self.pos_embed_drop(hidden_states)

        hidden_states = self.transformer(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=None,
            class_labels=None,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            hidden_states_is_embedding=True,
            unpatchify=False,
        )[0]

        if self.use_data_type_embedding:
            splits = (1, 1, 1, num_text_tokens, 1, num_img_tokens, num_img_tokens)
            (t_img_out, t_text_out, dt_out,
             text_out,
             img_clip_out, img_vae_out, label_out) = hidden_states.split(splits, dim=1)
        else:
            splits = (1, 1, num_text_tokens, 1, num_img_tokens, num_img_tokens)
            (t_img_out, t_text_out,
             text_out,
             img_clip_out, img_vae_out, label_out) = hidden_states.split(splits, dim=1)

        text_out = self.post_text(self.text_out(text_out))
        img_clip_out = self.clip_img_out(img_clip_out)

        img_vae_out = self.vae_img_out(img_vae_out)
        height = width = int(img_vae_out.shape[1] ** 0.5)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        img_vae_out = torch.einsum("nhwpqc->nchpwq", img_vae_out)
        img_vae_out = img_vae_out.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        label_out = self.label_out(label_out)
        label_out = label_out.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        label_out = torch.einsum("nhwpqc->nchpwq", label_out)
        label_out = label_out.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        return text_out, img_vae_out, img_clip_out, label_out
