import json
from pathlib import Path

p = Path(r"C:\Geochat\geochat\bigearthnet_1000\train_1000.json")
data = json.loads(p.read_text(encoding="utf-8-sig"))

for i, item in enumerate(data[:5]):
    print("\n" + "=" * 65)
    print("EXAMPLE:", i)
    print("IMAGE:", item.get("image", item.get("image_id", "NOT FOUND")))

    for message in item.get("conversations", []):
        print(f"{message.get('from', '?').upper()}:")
        print(message.get("value", "")[:1000])

