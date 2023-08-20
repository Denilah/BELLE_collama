model=QWen-7b-hf
ckpt_path=/hy-tmp/QWen_train/saved_models/QWen_code_7b
temp=0.2
max_len=512
pred_num=20
num_seqs_per_iter=5

output_path=preds/${model}-lora-1024_T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
gpu_num=1
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 164))
  end_index=$(((i + 1) * 164))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python generate_codet5p.py --model /hy-tmp/QWen_train/${model} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --use_lora --ckpt_path ${ckpt_path} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
