import json
import os
import random
from typing import List, Tuple

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class MUSEEnergyDataset10(BaseSegDataset):
    METAINFO = dict(
        classes=tuple(f"energy_class_{i}" for i in range(1, 10)),
        palette=[
            [49, 130, 189],
            [102, 194, 165],
            [255, 221, 84],
            [245, 130, 48],
            [215, 25, 28],
            [118, 42, 131],
            [53, 151, 143],
            [140, 81, 10],
            [197, 27, 125],
        ],
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix=".jpg",
            seg_map_suffix=".png",
            ignore_index=255,
            reduce_zero_label=True,
            **kwargs,
        )


def _landuse_climate(item: dict) -> Tuple[str, str]:
    full = item.get("prompt", "")
    parts = full.split("\n\n")
    landuse = parts[0] if parts else ""
    climate = parts[1] if len(parts) > 1 else item.get("climate_weather_text", "")
    return landuse, climate


@DATASETS.register_module()
class MUSEEnergyPromptDataset10(MUSEEnergyDataset10):
    VALID_MODES = ("correct", "shuffled", "null", "shuffled_climate_only")

    def __init__(self, prompt_jsonl: str, prompt_mode: str = "correct",
                 shuffle_seed: int = 0, **kwargs):
        assert prompt_mode in self.VALID_MODES, prompt_mode
        self.prompt_jsonl = prompt_jsonl
        self.prompt_mode = prompt_mode
        self.shuffle_seed = shuffle_seed
        self._texts: List[Tuple[str, str]] = self._load_prompt_texts(prompt_jsonl)
        super().__init__(**kwargs)

    @staticmethod
    def _load_prompt_texts(path: str) -> List[Tuple[str, str]]:
        texts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                texts.append(_landuse_climate(json.loads(line)))
        return texts

    def _perm(self, n: int) -> List[int]:
        idxs = list(range(n))
        random.Random(self.shuffle_seed).shuffle(idxs)
        return idxs

    def load_data_list(self):
        data_list = super().load_data_list()
        n = len(self._texts)
        perm = (self._perm(n)
                if self.prompt_mode in ("shuffled", "shuffled_climate_only")
                else None)
        for info in data_list:
            stem = os.path.splitext(os.path.basename(info["img_path"]))[0]
            try:
                idx = int(stem)
            except ValueError as e:
                raise ValueError(
                    f"image filename stem must be the integer JSONL line index, "
                    f"got non-integer stem '{stem}' from {info['img_path']}") from e
            assert idx < n, f"img idx {idx} >= prompt lines {n}"
            if self.prompt_mode == "null":
                landuse, climate = "", ""
            elif self.prompt_mode == "correct":
                landuse, climate = self._texts[idx]
            elif self.prompt_mode == "shuffled":
                landuse, climate = self._texts[perm[idx]]
            else:  # shuffled_climate_only
                landuse, _ = self._texts[idx]
                _, climate = self._texts[perm[idx]]
            info["landuse_text"] = landuse
            info["climate_text"] = climate
        return data_list
