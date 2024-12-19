"""
This file is to combine multi-thread generated soft probability example files in to one final file
The count for robin training data of block size 512 is around 31w, and for block size 2048 is 8w
"""


import os
import pandas
import sys
import json
import jsonlines
count = 0
for root, ds,fs in os.walk('/home/ksshumab/minrui/LMFlow-distill/raw'):
    for f in fs:
        # print(os.path.join(root,f))
        # if f.endswith(".jsonl"):
        list_a = []
        with open(os.path.join(root,f),"r+") as file:
            for item in jsonlines.Reader(file):
                list_a.append(item)
                if (len(list_a) % 1000 == 0):
                    print(f"{f} {len(list_a)} processed")

        output_writer = jsonlines.open("/home/ksshumab/minrui/Data-New/Alpaca/generated/alpaca_Qwen1.5-7B_finetune_2e-6_text2text.jsonl", "a")
        for index, output in enumerate(list_a):
            output_writer.write(output)
        count += len(list_a)
print(count)