import argparse
import json
import os
import sys
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.counterfactual import plan_tiles, split_prompt, ordered_months

DATA_ROOT = "data/MUSE2"
DEFAULT_SAVE = "output/dynamic_gen"
DEFAULT_TRAIN_JSONL = os.path.join(
    DATA_ROOT, "mixed_json_class10_july/train-4cities-class10-july.jsonl")

def gen_paths(save_dir, idx):
    return (os.path.join(save_dir, "image", f"{idx}.jpg"),
            os.path.join(save_dir, "label", f"{idx}.png"))


def load_train_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_idx"] = len(rows)
            rows.append(row)
    return rows


def tile_plan(save_dir, months, src_month, skip_existing):
    src = str(src_month).zfill(2)
    ordered = ordered_months(src, months.keys())  # baseline first, rest in calendar order

    def done(m):
        img, lbl = gen_paths(save_dir, months[m]["_idx"])
        return os.path.exists(img) and os.path.exists(lbl)

    pending = [m for m in ordered if not (skip_existing and done(m))]
    if not pending:
        return True, False, []
    save_baseline = src in pending  # baseline files missing -> need saving
    replace_months = [(m, months[m]["_idx"])
                      for m in ordered if m != src and m in pending]
    return False, save_baseline, replace_months


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--finetuned_model_path", required=True)
    p.add_argument("--pretrained_model_name_or_path",
                   default="inference/saved_pipeline/dysense")
    p.add_argument("--pretrained_label_vae_path", required=True)
    p.add_argument("--lightweight_label_vae", action="store_true")
    p.add_argument("--use_climate_text", action="store_true")
    p.add_argument("--remoteclip_checkpoint_path", default=None)
    p.add_argument("--train_jsonl", default=DEFAULT_TRAIN_JSONL)
    p.add_argument("--src_month", default="04")
    p.add_argument("--city", default=None)
    p.add_argument("--tile_id", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=8.0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default=DEFAULT_SAVE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()
    if not args.use_climate_text:
        p.error("--use_climate_text is required")
    if not args.remoteclip_checkpoint_path:
        p.error("--remoteclip_checkpoint_path is required with --use_climate_text.")
    return args


def load_dysense_pipeline(args, weight_dtype, device):
    import torch
    from diffusers import AutoencoderKL
    from pipelines.modeling_uvit import DySENSEModel
    from pipelines.pipeline_dysense import DySENSEPipeline
    unet = DySENSEModel.from_pretrained(args.finetuned_model_path, torch_dtype=weight_dtype)
    kwargs = {"pretrained_model_name_or_path": args.pretrained_model_name_or_path,
              "unet": unet, "torch_dtype": weight_dtype}
    if args.lightweight_label_vae:
        from pipelines.modeling_energy_vae import EnergyVAE
        kwargs["label_vae"] = EnergyVAE.from_pretrained(
            args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    else:
        kwargs["label_vae"] = AutoencoderKL.from_pretrained(
            args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    pipeline = DySENSEPipeline.from_pretrained(**kwargs).to(device)
    if args.use_climate_text:
        from pipelines.remote_clip import FrozenRemoteCLIPTextEncoder
        pipeline.climate_text_encoder = FrozenRemoteCLIPTextEncoder(
            arch="ViT-L-14", checkpoint_path=args.remoteclip_checkpoint_path,
        ).to(device, dtype=torch.float16)
    return pipeline


def main():
    import torch
    args = parse_args()
    device = torch.device(args.device)
    pipeline = load_dysense_pipeline(args, torch.float16, device)


    rows = load_train_rows(args.train_jsonl)
    samples = plan_tiles(rows, src_month=args.src_month, city=args.city,
                         tile_id=args.tile_id, limit=args.limit,
                         num_shards=args.num_shards, shard_index=args.shard_index)
    os.makedirs(os.path.join(args.save_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "label"), exist_ok=True)
    src_month = str(args.src_month).zfill(2)
    print(f"shard {args.shard_index}/{args.num_shards}: {len(samples)} tile(s)")

    def run_joint(landuse, climate, capture):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        kw = dict(mode="joint", prompt=[landuse],
                  num_inference_steps=args.num_inference_steps,
                  guidance_scale=args.guidance_scale, generator=gen,
                  ignore_label=0, use_color_map=False, capture_trajectory=capture)
        if args.use_climate_text:
            kw["climate_prompt"] = [climate]
        return pipeline(**kw)

    def run_replaced(landuse, climate, vae_traj, clip_traj):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        kw = dict(mode="joint", prompt=[landuse],
                  num_inference_steps=args.num_inference_steps,
                  guidance_scale=args.guidance_scale, generator=gen,
                  ignore_label=0, use_color_map=False,
                  replace_vae_latents=vae_traj, replace_clip_latents=clip_traj)
        if args.use_climate_text:
            kw["climate_prompt"] = [climate]
        return pipeline(**kw)

    def save_sample(sample, idx):
        img_path, lbl_path = gen_paths(args.save_dir, idx)
        sample.images[0].save(img_path)
        sample.labels[0].save(lbl_path)

    rows_done = 0
    for n, ((city, tile_id), months) in enumerate(samples):

        skip_tile, save_baseline, replace_months = tile_plan(
            args.save_dir, months, src_month, args.skip_existing)
        if skip_tile:
            print(f"[{n + 1}/{len(samples)}] skip {city}/{tile_id} (all months done)")
            continue
        print(f"[{n + 1}/{len(samples)}] {city}/{tile_id} "
              f"(baseline save={save_baseline}, replace={len(replace_months)})")
        src_landuse, src_climate = split_prompt(months[src_month]["prompt"])
        baseline = run_joint(src_landuse, src_climate, capture=True)
        if save_baseline:
            save_sample(baseline, months[src_month]["_idx"])
            rows_done += 1
        vae_traj, clip_traj = baseline.vae_trajectory, baseline.clip_trajectory
        for m, idx in replace_months:
            landuse, climate = split_prompt(months[m]["prompt"])
            save_sample(run_replaced(landuse, climate, vae_traj, clip_traj), idx)
            rows_done += 1

    manifest = {"src_month": src_month, "seed": args.seed,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps,
                "sampling_method": "trajectory_replace_full_gen",
                "replace_components": "img_vae + img_clip",
                "output": "full_res image+label, no downsample",
                "num_shards": args.num_shards, "shard_index": args.shard_index,
                "tiles_in_shard": len(samples), "rows_written": rows_done}
    with open(os.path.join(args.save_dir, f"manifest_shard_{args.shard_index}.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"shard {args.shard_index} done -> {args.save_dir} ({rows_done} rows)")


if __name__ == "__main__":
    main()
