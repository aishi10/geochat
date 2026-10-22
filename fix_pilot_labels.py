import json
from pathlib import Path
import pandas as pd

work = Path(r"C:\Users\maheit\Downloads\BigEarthNet_Work")
out = Path(r"C:\Geochat\geochat\bigearthnet_pilot")

metadata = pd.read_parquet(work / "BigEarthNet_metadata.parquet")
labels_by_id = dict(zip(metadata["patch_id"], metadata["labels"]))

json_path = out / "pilot.json"
with open(json_path, encoding="utf-8") as f:
    samples = json.load(f)

for sample in samples:
    labels = labels_by_id[sample["id"]]
    if isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    sample["conversations"][1]["value"] = (
        "BigEarthNet reference labels: " + ", ".join(labels)
    )

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print("Corrected labels for", len(samples), "examples")
print(samples[0]["conversations"][1]["value"])
