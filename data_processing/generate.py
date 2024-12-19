"""
After get the tokenized training data (block size of 512 or 2048)
Use this file to generate soft probabilities of teacher model

"""
import datetime
import json
import jsonlines
import torch

from accelerate import Accelerator
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM


def to_tokens_and_logprobs(model, tokenizer, input_texts, accelerator, input_atten_mask, input_labels):
    input_ids = torch.Tensor(input_texts).to(torch.int32).to(accelerator.device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    batch = []
    for i in range(0,len(input_texts)):
        tokens = []
        token_log_probs = []
        token_top_log_probs = []
        temp = {
                "text": input_texts[i],
                "attention_mask": input_atten_mask[i],
                "top_log_probs": token_top_log_probs,
        }

        for j in range(0,probs.shape[1]):
            temp_token = torch.topk(probs[i][j].flatten(), 5).indices.tolist() #[1,2,3,4,5]
            tokens.append(tokenizer.decode(temp_token[0]))
            token_log_probs.append(probs[i][j][temp_token[0]].item())
            temp_top_k_logprob = {}
            for k in range(0,len(temp_token)):
                temp_top_k_logprob[str(temp_token[k])] = probs[i][j][temp_token[k]].item()
            token_top_log_probs.append(temp_top_k_logprob)

        assert len(tokens) == len(token_log_probs) == len(token_top_log_probs) == probs.shape[1]
        batch.append(temp)
    return batch

def main():
    tokenizer = AutoTokenizer.from_pretrained("finetuned_teacher_model_path", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained("finetuned_teacher_model_path",
                                                 torch_dtype=torch.bfloat16,
                                                 offload_folder="offload",
                                                 offload_state_dict=True,
                                                 device_map="auto")
    accelerator = Accelerator()


    data = []
    with open("train_data_lath", "r+") as f:
        for item in jsonlines.Reader(f):
            data.append(item)

    batch_size = 1
    data = data[:10]
    starttime = datetime.datetime.now()

    for i in range(0,int(len(data)/batch_size)):
        input_texts = []
        input_atten_mask = []
        input_labels = []
        for j in range(0,batch_size):
            # text = tokenizer.decode(data[i]["input_ids"])
            # input_texts.append(text)
            input_texts.append(data[i*batch_size + j]["input_ids"])
            input_atten_mask.append(data[i*batch_size + j]["attention_mask"])
            input_labels.append(data[i*batch_size + j]["labels"])

        batch = to_tokens_and_logprobs(model, tokenizer, input_texts,accelerator, input_atten_mask, input_labels)
        output_writer = jsonlines.open("output_path", "a")
        for index, output in enumerate(batch):
            output_writer.write(output)
        nowtime = datetime.datetime.now()
        total_time = (nowtime-starttime).seconds
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"processed {i*batch_size + batch_size} / {len(data)}, Total time : {round(total_time/60,2)} min, still need  {round(total_time/60/(i*batch_size + batch_size)*(len(data)-(i*batch_size + batch_size)),2)} min")

    output_writer.close()

if __name__ == "__main__":
    main()