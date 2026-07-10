# Prepare the MMSegmentation data layout for downstream training.

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

TARGET_SIZE = (512, 512)


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process_real_item(idx, item, data_root, img_out, ann_out):
    fname_img = f"{idx:05d}.jpg"
    fname_ann = f"{idx:05d}.png"
    img_dst = os.path.join(img_out, fname_img)
    ann_dst = os.path.join(ann_out, fname_ann)
    if os.path.exists(img_dst) and os.path.exists(ann_dst):
        return

    sat_path = os.path.join(data_root, item["target"])
    with Image.open(sat_path) as img:
        img.convert("RGB").resize(TARGET_SIZE, resample=Image.BILINEAR).save(img_dst, quality=95)

    label_rel = item.get("energy_class_id_png", item.get("energy", ""))
    lbl_path = os.path.join(data_root, label_rel)
    with Image.open(lbl_path) as lbl:
        if lbl.mode != "L":
            lbl = lbl.convert("L")
        lbl.resize(TARGET_SIZE, resample=Image.NEAREST).save(ann_dst)


def prepare_real_split(json_path, split_name, data_root, mmseg_root, num_workers=16):
    img_out = os.path.join(mmseg_root, "img_dir", split_name)
    ann_out = os.path.join(mmseg_root, "ann_dir", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)

    items = list(iter_jsonl(json_path))
    print(f"\n[{split_name}] {len(items)} items from {json_path}")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(process_real_item, i, item, data_root, img_out, ann_out)
            for i, item in enumerate(items)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=split_name):
            fut.result()
    print(f"[{split_name}] done: {len(os.listdir(img_out))} pairs")


def prepare_synthetic(gen_dir, mmseg_root):
    src_img = os.path.join(gen_dir, "image")
    src_ann = os.path.join(gen_dir, "label")
    if not os.path.isdir(src_img):
        print(f"[SKIP train_synthetic] {src_img} not found - run generation first.")
        return

    img_out = os.path.join(mmseg_root, "img_dir", "train_synthetic")
    ann_out = os.path.join(mmseg_root, "ann_dir", "train_synthetic")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)

    src_files = sorted([f for f in os.listdir(src_img) if f.endswith(".jpg")])
    print(f"\n[train_synthetic] copying {len(src_files)} from {gen_dir}")

    missing = 0
    for fname in tqdm(src_files, desc="train_synthetic"):
        stem = os.path.splitext(fname)[0]
        dst_img_path = os.path.join(img_out, f"{int(stem):05d}.jpg")
        dst_ann_path = os.path.join(ann_out, f"{int(stem):05d}.png")

        img_src_path = os.path.join(src_img, fname)
        ann_src_path = os.path.join(src_ann, f"{stem}.png")
        if not os.path.exists(ann_src_path):
            ann_src_path = os.path.join(src_ann, f"{int(stem):05d}.png")

        if os.path.isfile(img_src_path):
            shutil.copy2(img_src_path, dst_img_path)
        else:
            missing += 1
            continue
        if os.path.isfile(ann_src_path):
            shutil.copy2(ann_src_path, dst_ann_path)
        else:
            missing += 1

    if missing:
        print(f"[WARN] {missing} synthetic files missing")
    print(f"[train_synthetic] done: img={len(os.listdir(img_out))}")


def print_summary(mmseg_root):
    print("\n" + "=" * 60)
    print(f"Prepared: {mmseg_root}")
    for sub in ("img_dir/train_real", "img_dir/train_synthetic", "img_dir/val"):
        p = os.path.join(mmseg_root, sub)
        n = len(os.listdir(p)) if os.path.isdir(p) else 0
        print(f"  {sub}: {n}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/MUSE2",
                        help="MUSE2 dataset root.")
    parser.add_argument("--train-jsonl", type=str, default=None,
                        help="Defaults to <data-root>/mixed_json_class10_july/train-4cities-class10-july.jsonl")
    parser.add_argument("--test-jsonl", type=str, default=None,
                        help="Defaults to <data-root>/mixed_json_class10_july/test-4cities-class10-july.jsonl")
    parser.add_argument("--mmseg-root", type=str, default="data/muse_mmseg_class10",
                        help="Output root of the MMSeg img_dir/ann_dir layout.")
    parser.add_argument("--gen-dir", type=str, default="output/dynamic_gen",
                        help="Generation output dir; used with --with-synthetic.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--with-synthetic", action="store_true",
                        help="Also prepare the synthetic split. Off by default "
                             "because generation may not have finished yet.")
    args = parser.parse_args()

    train_jsonl = args.train_jsonl or os.path.join(
        args.data_root, "mixed_json_class10_july/train-4cities-class10-july.jsonl")
    test_jsonl = args.test_jsonl or os.path.join(
        args.data_root, "mixed_json_class10_july/test-4cities-class10-july.jsonl")

    prepare_real_split(train_jsonl, "train_real", args.data_root, args.mmseg_root,
                       num_workers=args.num_workers)
    prepare_real_split(test_jsonl, "val", args.data_root, args.mmseg_root,
                       num_workers=args.num_workers)
    if args.with_synthetic:
        prepare_synthetic(args.gen_dir, args.mmseg_root)
    else:
        print("\n[train_synthetic] skipped (re-run with --with-synthetic after generation).")
    print_summary(args.mmseg_root)


if __name__ == "__main__":
    main()
