from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS

from pipelines.remote_clip import FrozenRemoteCLIPTextEncoder
from .text_fusion import fuse_text_into_memory

RC_HIDDEN = 768
TOKENS_PER_STREAM = 77


@MODELS.register_module()
class Mask2FormerMultimodalHead(Mask2FormerHead):

    def __init__(self, *args, remoteclip_ckpt: str,
                 remoteclip_trainable: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        feat_channels = kwargs["feat_channels"]

        self.remoteclip = FrozenRemoteCLIPTextEncoder(
            arch="ViT-L-14", checkpoint_path=remoteclip_ckpt,
            trainable=remoteclip_trainable)

        self.rc_proj = nn.Linear(RC_HIDDEN, feat_channels)
        self.num_text_tokens = TOKENS_PER_STREAM
        self.text_level_embed = nn.Parameter(torch.zeros(1, 1, feat_channels))
        self.text_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_text_tokens, feat_channels))

    def _encode_text(self, batch_data_samples) -> Tensor:
        device = self.rc_proj.weight.device
        dtype = self.rc_proj.weight.dtype
        climate = [ds.metainfo["climate_text"] for ds in batch_data_samples]
        rc_emb = self.remoteclip(climate)  # (B,77,768) on rc device
        return rc_emb.to(device=device, dtype=dtype)

    def forward(self, x: List[Tensor],
                batch_data_samples) -> Tuple[List[Tensor]]:
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        rc_emb = self._encode_text(batch_data_samples)
        text_tok = self.rc_proj(rc_emb)
        text_tok = text_tok + self.text_level_embed
        text_pos = self.text_pos_embed.expand(batch_size, -1, -1)

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            key, value, key_pos, cross_attn_mask = fuse_text_into_memory(
                decoder_inputs[level_idx],
                decoder_positional_encodings[level_idx],
                attn_mask, text_tok, text_pos)
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=key,
                value=value,
                query_pos=query_embed,
                key_pos=key_pos,
                cross_attn_mask=cross_attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list
