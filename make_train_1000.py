import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw

CSV = Path(r"C:\Users\maheit\Downloads\BigEarthNet_Work\BigEarthNet_GeoChat_pairs.csv")
OUT = Path(r"C:\Geochat\geochat\bigearthnet_1000")
IMG_DIR = OUT / "images"
TARGET = 1000

OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

def stretch(arr):
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        raise ValueError("Raster has no finite pixels")
    lo, hi = np.percentile(arr[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[valid] = np.clip((arr[valid] - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return out

def read_band(path):
    with rasterio.open(str(path)) as src:
        return src.read(1)

def parse_labels(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = [x.strip() for x in text.split(";") if x.strip()]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple, set)):
        return [str(x) for x in parsed]
    return [str(parsed)]

columns = [
    "patch_id", "vv_path", "vh_path",
    "b02_path", "b03_path", "b04_path", "labels", "split"
]
print("Reading paired CSV...")
df = pd.read_csv(CSV, usecols=columns)
df = df[df["split"].astype(str).str.lower().eq("train")]
print("Training rows available:", len(df))

examples = []
failures = 0

for _, row in df.iterrows():
    if len(examples) >= TARGET:
        break

    paths = [Path(str(row[c])) for c in ["vv_path", "b02_path", "b03_path", "b04_path"]]
    if not all(p.is_file() for p in paths):
        failures += 1
        continue

    try:
        vv = read_band(row["vv_path"])
        b02 = read_band(row["b02_path"])
        b03 = read_band(row["b03_path"])
        b04 = read_band(row["b04_path"])

        sar = Image.fromarray(stretch(vv)).convert("RGB")
        rgb = Image.fromarray(np.stack([stretch(b04), stretch(b03), stretch(b02)], axis=-1))

        # Side-by-side composite with clear modality headings.
        tile = 220
        sar = sar.resize((tile, tile), Image.Resampling.BILINEAR)
        rgb = rgb.resize((tile, tile), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (tile * 2, tile + 32), "white")
        canvas.paste(sar, (0, 32))
        canvas.paste(rgb, (tile, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 9), "Sentinel-1 SAR (VV)", fill="black")
        draw.text((tile + 8, 9), "Sentinel-2 RGB", fill="black")

        name = f"pair_{len(examples):05d}.jpg"
        canvas.save(IMG_DIR / name, quality=92)

        labels = parse_labels(row["labels"])
        if not labels:
            raise ValueError("No labels")

        examples.append({
            "image": f"images/{name}",
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
        failures += 1
        print("Skipping", row["patch_id"], ":", str(e)[:180])

if len(examples) != TARGET:
    raise RuntimeError(
        f"Created only {len(examples)} valid examples; expected {TARGET}. "
        "Fix the reported path or raster errors before training."
    )

json_path = OUT / "train_1000.json"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print("SUCCESS")
print("Examples:", len(examples))
print("Skipped rows:", failures)
print("JSON:", json_path)
print("Images:", IMG_DIR)
print("First example:", examples[0])
