#! /bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export WANDB_PROJECT=qwen
export WANDB_RUN_ID=7b_lora
export WANDB_RESUME=allow
model_name_or_path=/hy-tmp/QWen-7b-hf

train_file=/hy-tmp/data/MID_train_converted2d_EN_23K.json
validation_file=/hy-tmp/data/MID_val_converted2d_EN_2K_base.json
output_dir="/hy-tmp/Collama/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=2048

#FT
# torchrun --nproc_per_node 8 --master_port=25572 /hy-tmp/train/src/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --deepspeed configs/deepspeed_config.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 4e-5 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#    # --use_flash_attention
#    # --resume_from_checkpoint ...


#LoRA with 8bit
torchrun --nproc_per_node 4 --master_port=25572 /hy-tmp/Collama/src/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --use_lora \
    --use_int8_training \
    --lora_config /hy-tmp/Collama/configs/lora_config_QWen.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 4e-5 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    # --use_flash_attention
   # --resume_from_checkpoint ...

# LoRA without 8bit
# torchrun --nproc_per_node 4 /hy-tmp/train/src/sft_train.py \
#     --ddp_timeout 36000 \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora \
#     --deepspeed /hy-tmp/train/configs/deepspeed_config_stage3.json \
#     --lora_config /hy-tmp/train/configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 10 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 3e-4 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.01 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --torch_dtype "bfloat16" \
#     --bf16 \
#     --seed 1234 \
#     --gradient_checkpointing \
#     --cache_dir ${cache_dir} \
#     --output_dir ${output_dir} \
#     --use_flash_attention
   # --resume_from_checkpoint ...
