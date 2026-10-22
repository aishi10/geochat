import json

with open(r"C:\Geochat\Dataset\train_split.json") as f:
    data = json.load(f)

subset = data[:75000]

with open(r"C:\Geochat\Dataset\train_subset_75k.json", "w") as f:
    json.dump(subset, f)

print(f"Subset size: {len(subset)}")
