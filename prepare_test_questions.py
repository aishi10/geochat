import json
from pathlib import Path

root = Path(r"C:\Geochat\geochat\bigearthnet_test100")
source = root / "test_100.json"
questions_path = root / "questions.jsonl"
references_path = root / "references.json"

data = json.loads(source.read_text(encoding="utf-8"))
questions = []
references = []

for i, item in enumerate(data):
    questions.append({
        "question_id": str(i),
        "image": item["image"],
        "text": "Compare the Sentinel-1 SAR (VV) image on the left with the Sentinel-2 RGB image on the right. Identify all visible BigEarthNet land-cover classes. Respond with a comma-separated list of class names."
    })
    references.append({
        "question_id": str(i),
        "image": item["image"],
        "reference": item["conversations"][1]["value"].replace(
            "BigEarthNet reference labels: ", ""
        )
    })

with questions_path.open("w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

references_path.write_text(
    json.dumps(references, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Questions:", len(questions))
print("References:", len(references))
print("Questions file:", questions_path)
print("Reference file:", references_path)
