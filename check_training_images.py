from pathlib import Path
from PIL import Image
import json

root = Path(r"C:\Geochat\geochat")
data = json.loads(
    (root / "bigearthnet_1000" / "train_1000.json").read_text(encoding="utf-8-sig")
)

for item in data[:5]:
    rel = item.get("image", "")
    path = root / "bigearthnet_1000" / rel
    print("\nImage entry:", rel)
    print("Resolved path:", path)
    print("Exists:", path.exists())

    if path.exists():
        with Image.open(path) as im:
            print("Size:", im.size, "Mode:", im.mode)
