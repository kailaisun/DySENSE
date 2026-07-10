import json
import os
from pathlib import Path

import datasets
import numpy as np
from PIL import Image


class MUSE2Class10(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    TRAIN_JSONS = [Path("mixed_json_class10_july/train-4cities-class10-july.jsonl")]
    VAL_JSONS = [Path("mixed_json_class10_july/test-4cities-class10-july.jsonl")]

    ROOT = "data/MUSE2"

    TARGET_SIZE = (512, 512)  # (H, W)

    @property
    def category_names(self):
        return ["background"] + [f"energy_class_{i}" for i in range(1, 10)]

    @property
    def ignore_label(self):
        return 0

    @property
    def num_classes(self):
        return len(self.category_names) - 1  # 9

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "image": datasets.Image(),
                "semantic": datasets.Image(),
                "category_name_caption": datasets.Value("string"),
                "climate_caption": datasets.Value("string"),
                "meta": {
                    "city": datasets.Value("string"),
                    "file_name": datasets.Value("string"),
                    "label_path": datasets.Value("string"),
                    "month": datasets.Value("string"),
                },
            }),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"json_paths": [str(p) for p in self.TRAIN_JSONS]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"json_paths": [str(p) for p in self.VAL_JSONS]},
            ),
        ]

    def _iter_jsonl(self, json_path: str):
        full_path = os.path.join(self.ROOT, json_path)
        with open(full_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _resolve_path(self, rel_path: str) -> str:
        return os.path.join(self.ROOT, rel_path)

    def _load_satellite(self, item: dict) -> Image.Image:
        p = self._resolve_path(item["target"])
        with Image.open(p) as img:
            img = img.convert("RGB")
            img = img.resize(self.TARGET_SIZE[::-1], resample=Image.BILINEAR)
            return img.copy()

    def _load_energy_mask(self, item: dict) -> Image.Image:
        rel_path = item.get("energy_class_id_png", item.get("energy"))
        p = self._resolve_path(rel_path)
        with Image.open(p) as img:
            if img.mode != "L":
                img = img.convert("L")
            img = img.resize(self.TARGET_SIZE[::-1], resample=Image.NEAREST)
            return img.copy()

    def _generate_examples(self, json_paths):
        idx = 0
        for json_path in json_paths:
            for item in self._iter_jsonl(json_path):
                satellite = self._load_satellite(item)
                mask = self._load_energy_mask(item)
                city = item.get("city", "")
                month = item.get("month", "")
                target_path = item.get("target", "")
                file_name = Path(target_path).name
                label_rel = item.get("energy_class_id_png", item.get("energy", ""))
                label_path = self._resolve_path(label_rel)

                yield idx, {
                    "image": satellite,
                    "semantic": mask,
                    "category_name_caption": str(item.get("prompt", "")).split("\n\n")[0],
                    "climate_caption": str(item.get("prompt", "")).split("\n\n")[1] if "\n\n" in str(item.get("prompt", "")) else str(item.get("climate_weather_text", "")),
                    "meta": {
                        "city": str(city),
                        "file_name": file_name,
                        "label_path": label_path,
                        "month": str(month),
                    },
                }
                idx += 1
