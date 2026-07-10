import torch


def fuse_text_into_memory(img_mem, img_pos, attn_mask, text_tok, text_pos):
    assert attn_mask.shape[-1] == img_mem.shape[1], (
        f"attn_mask HW {attn_mask.shape[-1]} != img_mem HW {img_mem.shape[1]}")
    key = torch.cat([img_mem, text_tok], dim=1)
    key_pos = torch.cat([img_pos, text_pos], dim=1)
    bh, q, _ = attn_mask.shape
    t = text_tok.shape[1]
    text_cols = attn_mask.new_zeros((bh, q, t))  # bool False => attendable
    cross_attn_mask = torch.cat([attn_mask, text_cols], dim=-1)
    return key, key, key_pos, cross_attn_mask
