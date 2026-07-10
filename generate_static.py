import argparse
import json
import math
import os
import random

import torch
import wandb
from accelerate import PartialState
from diffusers import AutoencoderKL
from tqdm import tqdm

from pipelines.modeling_uvit import DySENSEModel
from pipelines.pipeline_dysense import DySENSEPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        required=True,
        help="Path to finetuned DySENSE UNet weights, "
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="inference/saved_pipeline/dysense")
    parser.add_argument("--pretrained_label_vae_path", type=str, required=True,
                        help="Path to the trained label VAE from Stage 1, e.g. output/energy-vae.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="muse2_class10")
    parser.add_argument("--lightweight_label_vae", action="store_true")
    parser.add_argument("--generate_mode", type=str, default="joint", choices=["text2img", "joint"])
    parser.add_argument("--num_images", type=int, default=0,
                        help="Number of images to generate. 0 = generate all prompts from dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_split", type=str, default="test", choices=["train", "test", "both"],
                        help="Which JSONL split to sample prompts from.")
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=1,
        help="Number of images per pipeline call. Increase to use more GPU memory.",
    )
    parser.add_argument("--use_climate_text", action="store_true",
                        help="Enable dual-prompt with RemoteCLIP climate text encoder.")
    parser.add_argument("--remoteclip_checkpoint_path", type=str, default=None,
                        help="Path to RemoteCLIP-ViT-L-14.pt checkpoint file.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="dysense-generation")
    parser.add_argument("--wandb_sample_interval", type=int, default=400,
                        help="Log a sample image to wandb every N generated images.")
    args = parser.parse_args()
    if args.use_climate_text and not args.remoteclip_checkpoint_path:
        parser.error("--remoteclip_checkpoint_path is required with --use_climate_text.")
    return args


def _flush_batch(pipeline, args, generator, batch_items):
    prompts = [item[1].split("\n\n")[0] for item in batch_items]
    climate_prompts = [item[2] for item in batch_items] if len(batch_items[0]) > 2 else None
    call_kwargs = dict(
        mode=args.generate_mode,
        prompt=prompts,
        num_inference_steps=50,
        generator=generator,
        ignore_label=0,
        use_color_map=False,
    )
    if climate_prompts is not None:
        call_kwargs["climate_prompt"] = climate_prompts
    sample = pipeline(**call_kwargs)
    for i, item in enumerate(batch_items):
        idx = item[0]
        img_path = f"{args.save_dir}/image/{idx}.jpg"
        lbl_path = f"{args.save_dir}/label/{idx}.png"
        sample.images[i].save(img_path)
        sample.labels[i].save(lbl_path)
    return sample

def generate(args, weight_dtype):
    random.seed(args.seed)
    unet = DySENSEModel.from_pretrained(args.finetuned_model_path, torch_dtype=weight_dtype)
    pipeline_kwargs = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "unet": unet, "torch_dtype": weight_dtype,
    }
    if args.lightweight_label_vae:
        from pipelines.modeling_energy_vae import EnergyVAE
        pipeline_kwargs["label_vae"] = EnergyVAE.from_pretrained(args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    else:
        pipeline_kwargs["label_vae"] = AutoencoderKL.from_pretrained(args.pretrained_label_vae_path, torch_dtype=weight_dtype)
    pipeline = DySENSEPipeline.from_pretrained(**pipeline_kwargs)
    pipeline.set_progress_bar_config(disable=True)

    if args.use_climate_text:
        from pipelines.remote_clip import FrozenRemoteCLIPTextEncoder
        climate_encoder = FrozenRemoteCLIPTextEncoder(
            arch="ViT-L-14", checkpoint_path=args.remoteclip_checkpoint_path,
        )
        pipeline.climate_text_encoder = climate_encoder

    distributed_state = PartialState()
    pipeline = pipeline.to(distributed_state.device)
    if hasattr(pipeline, 'climate_text_encoder') and pipeline.climate_text_encoder is not None:
        pipeline.climate_text_encoder = pipeline.climate_text_encoder.to(
            distributed_state.device, dtype=torch.float16,
        )
    generator = torch.Generator(device=distributed_state.device).manual_seed(args.seed)
    if distributed_state.is_main_process:
        if os.path.exists(f"{args.save_dir}/image") and os.listdir(f"{args.save_dir}/image"):
            print(f"Directory {args.save_dir}/image already exists; existing files will be skipped.")
        os.makedirs(f"{args.save_dir}/image", exist_ok=True)
        os.makedirs(f"{args.save_dir}/label", exist_ok=True)
        print(f"Generate args: {args}")

    # prepare dataset to args.num_images
    if args.dataset_name != "muse2_class10":
        raise ValueError(f"Unknown dataset {args.dataset_name}")

    climate_list = None
    if args.dataset_name == "muse2_class10":
        data_root = "data/MUSE2"
        jsonl_dir, jsonl_suffix = "mixed_json_class10_july", "class10-july"
        split_files = []
        if args.prompt_split in ["train", "both"]:
            split_files.append(os.path.join(data_root, f"{jsonl_dir}/train-4cities-{jsonl_suffix}.jsonl"))
        if args.prompt_split in ["test", "both"]:
            split_files.append(os.path.join(data_root, f"{jsonl_dir}/test-4cities-{jsonl_suffix}.jsonl"))
        caption_list = []
        if args.use_climate_text:
            climate_list = []
        for fp in split_files:
            with open(fp, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    prompt = item.get("prompt", "")
                    if prompt:
                        caption_list.append(prompt)
                    if climate_list is not None:
                        full_prompt = item.get("prompt", "")
                        if "\n\n" in full_prompt:
                            climate_list.append(full_prompt.split("\n\n")[1])
                        else:
                            climate_list.append(item.get("climate_weather_text", ""))

    if climate_list is not None:
        assert len(caption_list) == len(climate_list), (
            f"Prompt/climate list length mismatch: {len(caption_list)} vs {len(climate_list)}"
        )

    if args.num_images > 0 and args.num_images < len(caption_list):
        if climate_list is not None:
            paired = list(zip(caption_list, climate_list))
            paired = paired * math.ceil(args.num_images / len(paired))
            random.shuffle(paired)
            paired = paired[:args.num_images]
            caption_list = [p for p, _ in paired]
            climate_list = [c for _, c in paired]
        else:
            caption_list = caption_list * math.ceil(args.num_images / len(caption_list))
            random.shuffle(caption_list)
            caption_list = caption_list[:args.num_images]

    if climate_list is not None:
        caption_dataset = [[idx, c, cl] for idx, (c, cl) in enumerate(zip(caption_list, climate_list))]
    else:
        caption_dataset = [[idx, c] for idx, c in enumerate(caption_list)]

    total_images_all = len(caption_dataset)

    if args.use_wandb and distributed_state.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=f"gen-{os.path.basename(args.save_dir)}",
            config=vars(args),
        )
        wandb.define_metric("generated_images")
        wandb.define_metric("*", step_metric="generated_images")

    with distributed_state.split_between_processes(caption_dataset) as split_caption:
        batch_items = []
        generated_count = 0
        pbar = tqdm(split_caption, desc="generate")
        for item in pbar:
            idx = item[0]
            img_path = f"{args.save_dir}/image/{idx}.jpg"
            lbl_path = f"{args.save_dir}/label/{idx}.png"
            if os.path.isfile(img_path) and os.path.isfile(lbl_path):
                continue
            batch_items.append(tuple(item))
            if len(batch_items) >= args.gen_batch_size:
                sample = _flush_batch(pipeline, args, generator, batch_items)
                generated_count += len(batch_items)

                if args.use_wandb and distributed_state.is_main_process:
                    global_generated = generated_count * distributed_state.num_processes
                    log_data = {
                        "generated_images": global_generated,
                        "progress_pct": global_generated / total_images_all * 100,
                    }
                    prev_milestone = (global_generated - len(batch_items) * distributed_state.num_processes) // args.wandb_sample_interval
                    curr_milestone = global_generated // args.wandb_sample_interval
                    if curr_milestone > prev_milestone:
                        clip_prompt = batch_items[0][1].split("\n\n")[0]
                        climate_text = batch_items[0][2] if len(batch_items[0]) > 2 else ""
                        log_data["sample/image"] = wandb.Image(
                            sample.images[0], caption=clip_prompt)
                        log_data["sample/label"] = wandb.Image(
                            sample.labels[0], caption="Generated Label")
                        log_data["sample/prompt"] = clip_prompt
                        log_data["sample/climate_prompt"] = climate_text
                    wandb.log(log_data)

                batch_items = []
        if batch_items:
            sample = _flush_batch(pipeline, args, generator, batch_items)
            generated_count += len(batch_items)
            if args.use_wandb and distributed_state.is_main_process:
                global_generated = generated_count * distributed_state.num_processes
                wandb.log({
                    "generated_images": global_generated,
                    "progress_pct": global_generated / total_images_all * 100,
                })

    if args.use_wandb and distributed_state.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    generate(parse_args(), torch.float16)
