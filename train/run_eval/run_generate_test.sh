model=QWen-7b-hf
temp=0.2
max_len=1024
pred_num=20
num_seqs_per_iter=10
ckpt_path=/hy-tmp/QWen_train/saved_models/QWen_code_7b

output_path=preds/${model}_T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model


CUDA_VISIBLE_DEVICES=0 python generate_test.py --model /hy-tmp/QWen_train/${model} \
      --start_index 0 --end_index 4 --temperature ${temp} \
      --use_lora --ckpt_path ${ckpt_path}
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path}
