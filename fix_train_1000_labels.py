import json
import re
from pathlib import Path
import pandas as pd

csv_path = Path(r"C:\Users\maheit\Downloads\BigEarthNet_Work\BigEarthNet_GeoChat_pairs.csv")
json_path = Path(r"C:\Geochat\geochat\bigearthnet_1000\train_1000.json")

df = pd.read_csv(csv_path, usecols=["labels", "split"])
df = df[df["split"].astype(str).str.lower().eq("train")].head(1000)

with json_path.open("r", encoding="utf-8-sig") as f:
    examples = json.load(f)

if len(df) != 1000 or len(examples) != 1000:
    raise RuntimeError(f"Count mismatch: CSV={len(df)}, JSON={len(examples)}")

for row, example in zip(df.itertuples(index=False), examples):
    labels = re.findall(r"'([^']*)'", str(row.labels))
    if not labels:
        raise ValueError(f"Could not parse labels: {row.labels!r}")
    example["conversations"][1]["value"] = "BigEarthNet reference labels: " + ", ".join(labels)

with json_path.open("w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print("Corrected label formatting for", len(examples), "examples.")
print("First corrected answer:")
print(examples[0]["conversations"][1]["value"])
print("Image count:", len(list(json_path.parent.joinpath("images").glob("*.jpg"))))
