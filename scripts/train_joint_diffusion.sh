set -e

accelerate launch train_joint_diffusion.py \
  --pretrained_model_name_or_path inference/saved_pipeline/dysense \
  --pretrained_label_vae_path output/energy_vae \
  --lightweight_label_vae \
  --use_climate_text --remoteclip_checkpoint_path checkpoints/RemoteCLIP-ViT-L-14.pt \
  --output_dir output/joint_diffusion
