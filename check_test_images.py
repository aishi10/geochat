from pathlib import Path
from PIL import Image, ImageChops
import hashlib

root = Path(r"C:\Geochat\geochat\bigearthnet_test100\images")
files = [root / "test_000.jpg", root / "test_001.jpg", root / "test_002.jpg"]

for p in files:
    print("\n", p.name, "exists:", p.exists())
    if p.exists():
        with Image.open(p) as im:
            print("Size:", im.size, "Mode:", im.mode)
            print("SHA256:", hashlib.sha256(p.read_bytes()).hexdigest())

if all(p.exists() for p in files):
    with Image.open(files[0]) as a, Image.open(files[1]) as b:
        print("\nFirst two images pixel-identical:", ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getbbox() is None)
