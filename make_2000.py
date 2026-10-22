import json
import random

INPUT_FILE= r"C:\Geochat\Dataset\GeoChat_Instruct_clean.json"
OUTPUT_FILE= r"C:\Geochat\Dataset\test_2000.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data=json.load(f)

print(f"Full dataset: {len(data)} samples")

if len(data)<2000:
    raise ValueError(f"Dataset only has {len(data)} samples.")

random.seed(42)
subset=random.sample(data,2000)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(subset,f,ensure_ascii=False, indent=2)

print(f"Created:{OUTPUT_FILE}")
print(f"Subset size: {len(subset)} samples")