import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(r"C:\gc_weights\llava-v1.5-7b")

with open(r"C:\Geochat\Dataset\train_split.json") as f:
    data = json.load(f)

import random
random.seed(1)
sample = random.sample(data, min(2000, len(data)))

lengths = []
for item in sample:
    text = ""
    for turn in item.get("conversations", []):
        text += turn.get("value", "") + " "
    lengths.append(len(tokenizer.encode(text)))

lengths.sort()
n = len(lengths)
print(f"Sampled: {n}")
print(f"Min: {lengths[0]}, Max: {lengths[-1]}")
print(f"Median: {lengths[n//2]}")
print(f"90th percentile: {lengths[int(n*0.9)]}")
print(f"95th percentile: {lengths[int(n*0.95)]}")
print(f"99th percentile: {lengths[int(n*0.99)]}")
