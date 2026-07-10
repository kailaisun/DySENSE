from typing import Optional

import torch
import torch.nn as nn
import open_clip


class FrozenRemoteCLIPTextEncoder(nn.Module):

    LAYERS = ["last", "penultimate"]

    def __init__(
        self,
        arch: str = "ViT-L-14",
        checkpoint_path: Optional[str] = None,
        max_length: int = 77,
        layer: str = "last",
        trainable: bool = False,
    ):
        super().__init__()
        assert layer in self.LAYERS

        model, _, _ = open_clip.create_model_and_transforms(arch)
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            if isinstance(ckpt, dict):
                unwrapped = {}
                for k, v in ckpt.items():
                    unwrapped[k.removeprefix("module.")] = v
                ckpt = unwrapped
            model.load_state_dict(ckpt, strict=False)
        del model.visual
        self.model = model

        self.max_length = max_length
        self.layer_idx = 0 if layer == "last" else 1
        self._hidden_size = self._infer_hidden_size()

        if not trainable:
            self.freeze()
        else:
            self._freeze_unused_forward_params()

    def _freeze_unused_forward_params(self) -> None:
        for name in ("text_projection", "logit_scale"):
            param = getattr(self.model, name, None)
            if isinstance(param, torch.Tensor):
                param.requires_grad_(False)

    def _infer_hidden_size(self) -> int:
        return self.model.token_embedding.weight.shape[1]

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def max_position_embeddings(self) -> int:
        return self.max_length

    @property
    def dtype(self):
        return self.model.ln_final.weight.dtype

    @property
    def device(self):
        return self.model.ln_final.weight.device

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text).to(self.device)
        return self.encode_with_transformer(tokens)

    def encode_with_transformer(self, text_tokens: torch.Tensor) -> torch.Tensor:
        x = self.model.token_embedding(text_tokens)
        x = x + self.model.positional_embedding
        x = self._text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)
        return x

    def _text_transformer_forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
