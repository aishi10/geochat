import json

data_path = r'C:\Geochat\Dataset\GeoChat_Instruct.json'
bad_indices = sorted([249435, 262179], reverse=True)

with open(data_path, 'r') as f:
    data = json.load(f)

for idx in bad_indices:
    removed = data.pop(idx)
    print(f"Removed index {idx}, image: {removed.get('image')}")

print(f"New total: {len(data)}")

with open(r'C:\Geochat\Dataset\GeoChat_Instruct_clean.json', 'w') as f:
    json.dump(data, f)
