from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.utils import get_uncertain_point_coords_with_randomness, point_sample


class SegmentationLosses(nn.Module):
    def __init__(
        self,
        num_points = 12544,
        oversample_ratio = 3,
        importance_sample_ratio = 0.75,
        ignore_label = 0,
        cost_mask = 1.0,
        cost_class = 1.0,
        temperature = 1.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_label = ignore_label
        self.temperature = temperature
        self.cost_mask = cost_mask
        self.cost_class = cost_class

    @torch.no_grad()
    def matcher(self, outputs, targets, pred_logits=None):
        bs = len(outputs)
        num_queries = outputs.shape[1]
        indices = []

        for b in range(bs):
            out_mask = outputs[b]
            tgt_mask = targets[b]['masks']
            if tgt_mask is None:
                indices.append(None)
                continue

            cost_class = 0
            if pred_logits is not None:
                tgt_ids = targets[b]["labels"]
                out_prob = pred_logits[b].softmax(-1)  # [num_queries, num_classes]
                cost_class = -out_prob[:, tgt_ids]
                cost_class = -out_prob.view(-1, 1)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with torch.cuda.amp.autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = self.matcher_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = self.matcher_dice_loss(out_mask, tgt_mask)

            C = self.cost_mask * (cost_mask + cost_dice) + self.cost_class * cost_class
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_masks(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        indices=None,
    ) -> torch.tensor:
        if indices is None:
            targets, indices = self.prepare_targets(targets, ignore_label=self.ignore_label)

        masks = [t["masks"] for t in targets]
        valids = [m is not None for m in masks]
        outputs = outputs[valids]
        indices = [idx for idx, v in zip(indices, valids) if v]
        masks = [m for m in masks if m is not None]

        num_masks = sum(len(m) for m in masks)
        if num_masks == 0:
            return outputs.sum() * 0.0
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=outputs.device)
        num_masks = torch.clamp(num_masks, min=1).item()

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs[src_idx]
        target_masks = torch.cat([t[idx[1]] for t, idx in zip(masks, indices)])

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        del src_masks
        del target_masks

        loss_mask = self.sigmoid_ce_loss(point_logits, point_labels, num_masks)
        loss_dice = self.dice_loss(point_logits, point_labels, num_masks)
        return loss_mask + loss_dice

    def dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    def matcher_dice_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        return loss.mean(1).sum() / num_masks

    def matcher_sigmoid_ce_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )

        return loss / hw

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))

    def calculate_uncertainty_seg(self, sem_seg_logits):
        top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
        return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)  # [B, 1, P]

    def loss_ce(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if indices is not None:
            src_masks = outputs
            masks = [t["masks"] for t in targets]
            src_idx = self._get_src_permutation_idx(indices)
            target_masks = torch.cat([t[idx[1]] for t, idx in zip(masks, indices)])
            target_masks = target_masks.bool()

            h, w = target_masks.shape[-2:]
            ce_targets = torch.full((len(targets), h, w), self.ignore_label, dtype=torch.int64, device=src_masks.device)
            for mask_idx, (x, y) in enumerate(zip(*src_idx)):
                mask_i = target_masks[mask_idx]
                ce_targets[x, mask_i] = y
            targets = ce_targets

        if masks is not None:
            targets[~masks[:, 0].bool()] = self.ignore_label

        if len(torch.unique(targets)) == 1:
            return outputs.sum() * 0.0

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                outputs,
                lambda logits: self.calculate_uncertainty_seg(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = point_sample(
                targets.unsqueeze(1).float(),
                point_coords,
                mode='nearest',
                align_corners=False,
            ).squeeze(1).to(torch.long)

        point_logits = point_sample(
            outputs,
            point_coords,
            align_corners=False,
        )

        ce_loss = F.cross_entropy(
            point_logits / self.temperature,
            point_labels,
            reduction="mean",
            ignore_index=self.ignore_label,
        )

        return ce_loss

    def point_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        do_matching: bool = False,
        masks: Optional[torch.Tensor] = None,
    ):
        indices = None
        if do_matching:
            targets, _ = self.prepare_targets(targets, ignore_label=self.ignore_label)
            indices = self.matcher(outputs, targets)

        ce_loss = self.loss_ce(outputs, targets, indices=indices, masks=masks)

        mask_loss = self.loss_masks(outputs, targets, indices=indices)
        losses = {'ce': ce_loss, 'mask': mask_loss}

        return losses

    @torch.no_grad()
    def prepare_targets(
        self,
        targets,
        ignore_label=0,
    ):
        new_targets = []
        instance_ids = []

        for idx_t, target in enumerate(targets):
            unique_classes = torch.unique(target)
            masks = []

            for idx in unique_classes:
                if idx == ignore_label:
                    continue
                binary_target = torch.where(target == idx, 1, 0).to(torch.float32)
                masks.append(binary_target)

            unique_classes_excl = unique_classes[unique_classes != ignore_label]
            instance_ids.append(
                (
                    unique_classes_excl.cpu(),
                    torch.arange(len(unique_classes_excl), dtype=torch.int64)
                )
            )

            new_targets.append(
                {
                    "labels": torch.full(
                        (len(masks),), 0, dtype=torch.int64, device=target.device) if len(masks) > 0 else None,
                    "masks": torch.stack(masks) if len(masks) > 0 else None,
                }
            )
        return new_targets, instance_ids
