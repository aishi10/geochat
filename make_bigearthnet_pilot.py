from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageOps

WORK = Path(r"C:\Users\maheit\Downloads\BigEarthNet_Work")
OUT = Path(r"C:\Geochat\geochat\bigearthnet_pilot")
IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(WORK / "BigEarthNet_GeoChat_pairs.csv").head(10)

def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)

def scale_percentile(a, low, high):
    valid = a[np.isfinite(a)]
    lo, hi = np.percentile(valid, [low, high])
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

def optical_rgb(row):
    # RGB display order: B04 red, B03 green, B02 blue.
    bands = [read_band(row[k]) for k in ("b04_path", "b03_path", "b02_path")]
    rgb = np.stack([scale_percentile(b, 2, 98) for b in bands], axis=-1)
    return Image.fromarray(rgb, "RGB")

def sar_rgb(row):
    vv = read_band(row["vv_path"])
    vh = read_band(row["vh_path"])
    # Display-only contrast scaling for SAR; not a calibrated model input.
    vv8 = scale_percentile(vv, 2, 98)
    vh8 = scale_percentile(vh, 2, 98)
    contrast = scale_percentile(vv - vh, 2, 98)
    return Image.fromarray(np.stack([vv8, vh8, contrast], axis=-1), "RGB")

records = []
for i, row in df.iterrows():
    sar = sar_rgb(row)
    optical = optical_rgb(row)

    # Label the two panels so their modalities are unambiguous.
    panel_w, panel_h = 240, 260
    canvas = Image.new("RGB", (panel_w * 2, panel_h), "white")
    sar = ImageOps.contain(sar, (panel_w, 230))
    optical = ImageOps.contain(optical, (panel_w, 230))
    canvas.paste(sar, ((panel_w - sar.width) // 2, 25))
    canvas.paste(optical, (panel_w + (panel_w - optical.width) // 2, 25))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), "SAR: VV / VH / VV-VH", fill="black")
    draw.text((panel_w + 8, 5), "Sentinel-2 RGB", fill="black")

    filename = f"pair_{i:05d}.jpg"
    canvas.save(IMG / filename, quality=95)

    labels = row["labels"]
    if isinstance(labels, str):
        labels = labels.strip("[]").replace("'", "").split()
    if isinstance(labels, str):
        labels = [labels]

    records.append({
        "id": str(row["patch_id"]),
        "image": f"images/{filename}",
        "conversations": [
            {
                "from": "human",
                "value": "Compare the paired Sentinel-1 SAR image on the left with the Sentinel-2 RGB image on the right. Identify the land-cover classes visible in the images."
            },
            {
                "from": "gpt",
                "value": "BigEarthNet reference labels: " + ", ".join(labels)
            }
        ]
    })

with open(OUT / "pilot.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Pilot images:", len(records))
print("JSON:", OUT / "pilot.json")
print("Images:", IMG)
