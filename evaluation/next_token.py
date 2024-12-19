import argparse
import datetime
import json
import jsonlines
import torch
import numpy as np
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM

def arg_parser():
    parser = argparse.ArgumentParser(description="LLM-Distill")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--dataset_path", type=str)
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()    
    print("--- Start Loading Model ---")
    print(f"The model is {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 offload_folder="offload",
                                                 offload_state_dict=True,
                                                 device_map="auto")

    print("--- Start Generating Probability ---")
    print(f"The datset is {args.dataset_path}")
    starttime = datetime.datetime.now()
    with open(args.dataset_path, "r+") as data_file:
        data_obj = json.loads(data_file.read())
        data_used = data_obj["instances"]
        total_len = len(data_used)

        acc = 0
        for i, input_text in enumerate(data_used):

            tokenized_text = tokenizer.encode(input_text['input'])
            tokenized_text_tensor = torch.Tensor([tokenized_text]).to(torch.int32).to(model.device)
            outputs = model(tokenized_text_tensor)
            probs = torch.softmax(outputs.logits[0, -1, :], dim=-1).detach() # [1, 512, 32000] -> [32000]

            top5_tokens = torch.topk(probs, 5).indices.tolist()
            top5_probs = torch.topk(probs, 5).values.tolist()

            # acc
            # llama2
            choice_map = {
                "a": 29874, "b": 29890, "c": 29883, "d": 29881, "e": 29872,
                "A": 29909, "B": 29933, "C": 29907, "D": 29928, "E": 29923,
                "no": 694, "yes": 4874,
            }
            # Qwen1.5
            choice_map = {
                "a": 64, "b": 65, "c": 66, "d": 67, "e": 68
            }
            ground_truth = choice_map[input_text['output'][0]]
            if(top5_tokens[0] == ground_truth):
                acc += 1

            temp = {
                    "top5_probs": top5_probs,
                    "top5_tokens": top5_tokens,
                    "top5_words": tokenizer.decode(top5_tokens),
                    "acc": acc/(i+1)
            }

            output_writer = jsonlines.open(args.output_dir, "a")
            output_writer.write(temp)
            nowtime = datetime.datetime.now()
            total_time = (nowtime-starttime).seconds
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"{i+1}/{total_len}, passed {round(total_time/60, 2)} mins, still need {round(total_time/60*(total_len-i)/(i+1),2)} mins")
            print(f"current acc = {acc/(i+1)}")
    output_writer.close()

if __name__ == "__main__":
    main()