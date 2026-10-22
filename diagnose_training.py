import json
from collections import Counter
from pathlib import Path

p = Path(r"C:\Geochat\geochat\bigearthnet_1000\train_1000.json")
data = json.loads(p.read_text(encoding="utf-8-sig"))

answers = []
class_counts = Counter()

for item in data:
    conv = item.get("conversations", [])
    answer = next(
        (m["value"] for m in conv if m.get("from") == "gpt"),
        ""
    )
    answers.append(answer.strip())

    for label in [
        "Arable land",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Pastures",
        "Inland waters",
        "Complex cultivation patterns",
        "Industrial or commercial units",
        "Urban fabric",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
    ]:
        if label in answer:
            class_counts[label] += 1

print("Training examples:", len(data))
print("\nClass frequency:")
for label, count in class_counts.most_common():
    print(f"{label}: {count} ({count/len(data):.1%})")

print("\nMost common complete answers:")
for answer, count in Counter(answers).most_common(10):
    print(f"{count:4d}  {answer}")
