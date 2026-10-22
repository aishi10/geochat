import json, random

with open(r"C:\Geochat\Dataset\GeoChat_Instruct_clean.json") as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

val_size = max(1, int(len(data) * 0.01))
val, train = data[:val_size], data[val_size:]

with open(r"C:\Geochat\Dataset\train_split.json", "w") as f:
    json.dump(train, f)
with open(r"C:\Geochat\Dataset\val_split.json", "w") as f:
    json.dump(val, f)

print(f"Train: {len(train)}, Val: {len(val)}")
