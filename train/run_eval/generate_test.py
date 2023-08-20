import argparse
import pprint
import os
import re
from tqdm import tqdm
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from human_eval.data import write_jsonl, read_problems, stream_jsonl


def extract_text(prompt, remove_lines=True):
    token = '\"\"\"'
    start = token
    end = '>>>'

    start_idx = prompt.find(start) + len(start)
    end_idx = prompt.find(end)

    output = prompt[start_idx: end_idx]
    if remove_lines:
        output = output.replace('\n', ' ')
    output = re.sub(r"\s+", " ", output).strip()

    return output


INSTRUCTION = """Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{}

### Response:"""



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Salesforce/instructcodet5p-16b', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=600, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--use_lora', action="store_true")
    parser.add_argument('--llama', action="store_true")
    parser.add_argument('--overwrite', action='store_true', help='')
    parser.add_argument('--ckpt_path', type=str, required=True)
    

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))
    
     if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model)
        model = LlamaForCausalLM.from_pretrained(args.model,
                                                 trust_remote_code=True,  
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, pad_token='<|endoftext|>', eos_token='<|endoftext|>')
        if args.use_lora:
            print("------------------------Use Lora！！---------------------")
            base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=load_type, trust_remote_code=True)
            model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type, trust_remote_code=True)


    model.eval()
    model.to(device)

    # for larger LLMs such as 2B, 6B, and 16B, we need to pass the text prompt to the decoder
    prompt_to_decoder = True if any([size in args.model for size in ['2b', '7b', '16b']]) else False

    print(f"Loaded {args.model}.")
    for i in tqdm(range(1), ncols=0, total=1):
        # output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)
        #
        # if os.path.exists(output_file) and not args.overwrite:
        #     print(f'Skip {output_file} as it already exists')
        #     continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch_decoder = [INSTRUCTION.format(extract_text(prompt)) + prompt]
        print("Prompt_batch_decoder:", prompt_batch_decoder)

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)


        with torch.no_grad():

            gen_tokens = model.generate(**encoding_decoder,
                                        do_sample=True,
                                        temperature=args.temperature,
                                        max_length=args.max_len,
                                        num_return_sequences=args.num_seqs_per_iter,
                                        decoder_start_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        top_p=0.95)


        if gen_tokens is not None:
            gen_tokens = gen_tokens[:, encoding_decoder['input_ids'].shape[-1]:]
            gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        else:
            gen_seqs = None

        if gen_seqs is not None:
            assert len(ids_batch) == 1
            task_id = ids_batch[0]

            for seq_idx, gen_seq in enumerate(gen_seqs):
                completion_seq = gen_seq
                for stop_seq in STOP_SEQS:
                    index = completion_seq.find(stop_seq)
                    if index != -1:
                        completion_seq = completion_seq[:index]
                completion_seq = completion_seq.replace('\t', '    ')
                all_code = prompt.replace('\t', '    ') + completion_seq

                completion_seqs.append(
                    {'task_id': task_id,
                     'completion': completion_seq,
                     'all_code': all_code  # final code for evaluation with unit tests
                     }
                )

        # print("Saving results to {}".format(output_file))
        # write_jsonl(output_file, completion_seqs)
        for d in completion_seqs:
            print(d)

if __name__ == '__main__':
    main()
