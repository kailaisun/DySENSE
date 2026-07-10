set -e

python generate_dynamic.py \
  --finetuned_model_path output/joint_diffusion/unet \
  --pretrained_label_vae_path output/energy_vae \
  --lightweight_label_vae \
  --use_climate_text --remoteclip_checkpoint_path checkpoints/RemoteCLIP-ViT-L-14.pt \
  --save_dir output/dynamic_gen
