set -e

accelerate launch train_energy_vae.py \
  --dataset_name muse2_class10 \
  --lightweight_label_vae \
  --output_dir output/energy_vae
