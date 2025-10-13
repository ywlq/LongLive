export CUDA_VISIBLE_DEVICES=1
torchrun \
  --nproc_per_node=1 \
  --master_port=29511 \
  inference.py \
  --config_path configs/longlive_inference.yaml