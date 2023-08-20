output_path=/hy-tmp/human-eval/run_eval/preds/QWen-7b-hf-lora-1024_T0.8_N20

# echo 'Output path: '$output_path
python process_preds.py --path ${output_path} --out_path ${output_path}.jsonl

evaluate_functional_correctness ${output_path}.jsonl