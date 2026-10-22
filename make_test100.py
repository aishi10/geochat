import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw

csv_path = Path(r"C:\Users\maheit\Downloads\BigEarthNet_Work\BigEarthNet_GeoChat_pairs.csv")
out_dir = Path(r"C:\Geochat\geochat\bigearthnet_test100")
img_dir = out_dir / "images"
json_path = out_dir / "test_100.json"

df = pd.read_csv(csv_path, usecols=[
    "patch_id", "vv_path", "b02_path", "b03_path", "b04_path", "labels", "split"
])
df = df[df["split"].astype(str).str.lower() == "test"].head(100).reset_index(drop=True)

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    arr[~np.isfinite(arr)] = 0
    valid = arr[arr > 0]
    if valid.size:
        lo, hi = np.percentile(valid, [2, 98])
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        else:
            arr = np.zeros_like(arr)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255).astype(np.uint8)

def parse_labels(value):
    return re.findall(r"'([^']*)'", str(value))

img_dir.mkdir(parents=True, exist_ok=True)
examples = []

for i, row in df.iterrows():
    try:
        sar = read_band(row["vv_path"])
        red = read_band(row["b04_path"])
        green = read_band(row["b03_path"])
        blue = read_band(row["b02_path"])

        sar_img = Image.fromarray(sar).convert("RGB")
        rgb_img = Image.fromarray(np.stack([red, green, blue], axis=-1)).convert("RGB")
        sar_img.thumbnail((220, 220))
        rgb_img.thumbnail((220, 220))

        canvas = Image.new("RGB", (470, 260), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), "Sentinel-1 SAR (VV)", fill="black")
        draw.text((245, 7), "Sentinel-2 RGB", fill="black")
        canvas.paste(sar_img, (10, 30))
        canvas.paste(rgb_img, (245, 30))

        filename = f"test_{i:03d}.jpg"
        canvas.save(img_dir / filename, quality=92)

        labels = parse_labels(row["labels"])
        if not labels:
            raise ValueError(f"Could not parse labels for {row['patch_id']}")

        examples.append({
            "image": f"images/{filename}",
            "conversations": [
                {
                    "from": "human",
                    "value": "Compare the Sentinel-1 SAR (VV) image on the left with the Sentinel-2 RGB image on the right. Identify the reference land-cover classes."
                },
                {
                    "from": "gpt",
                    "value": "BigEarthNet reference labels: " + ", ".join(labels)
                }
            ]
        })
    except Exception as e:
        print(f"ERROR on test row {i}: {e}")

with json_path.open("w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print("Created examples:", len(examples))
print("Images saved:", len(list(img_dir.glob("*.jpg"))))
print("JSON:", json_path)
if examples:
    print("First reference:", examples[0]["conversations"][1]["value"])
