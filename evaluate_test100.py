import json
import re
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

root = Path(r"C:\Geochat\geochat\bigearthnet_test100")
csv_path = Path(r"C:\Users\maheit\Downloads\BigEarthNet_Work\BigEarthNet_GeoChat_pairs.csv")
pred_path = root / "predictions.jsonl"

predictions = [
    json.loads(line)
    for line in pred_path.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
predictions.sort(key=lambda x: int(x["question_id"]))

df = pd.read_csv(csv_path)
test_df = df[df["split"].astype(str).str.lower() == "test"].head(100).reset_index(drop=True)

def parse_labels(value):
    return set(re.findall(r"'([^']*)'", str(value)))

if len(predictions) != 100 or len(test_df) != 100:
    raise RuntimeError(f"Expected 100 each; got {len(predictions)} predictions and {len(test_df)} rows")

true_sets = [parse_labels(v) for v in test_df["labels"]]
vocabulary = sorted(set().union(*true_sets))
vocabulary.sort(key=len, reverse=True)

def detect_labels(answer):
    answer = str(answer).lower()
    return {label for label in vocabulary if label.lower() in answer}

y_true, y_pred = [], []
details = []
exact = 0

for i, (true_labels, pred) in enumerate(zip(true_sets, predictions)):
    answer = pred.get("text", pred.get("answer", ""))
    predicted_labels = detect_labels(answer)
    y_true.append([int(label in true_labels) for label in vocabulary])
    y_pred.append([int(label in predicted_labels) for label in vocabulary])
    exact += int(true_labels == predicted_labels)
    details.append({
        "question_id": i,
        "reference_labels": " | ".join(sorted(true_labels)),
        "predicted_labels": " | ".join(sorted(predicted_labels)),
        "raw_answer": answer
    })

print("Test images:", len(predictions))
print("Distinct reference classes:", len(vocabulary))
print("Classes:", vocabulary)
print(f"Micro precision: {precision_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
print(f"Micro recall:    {recall_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
print(f"Micro F1:        {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
print(f"Macro F1:        {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
print(f"Exact match:     {exact / len(predictions):.4f} ({exact}/{len(predictions)})")

out = root / "evaluation_details.csv"
pd.DataFrame(details).to_csv(out, index=False, encoding="utf-8-sig")
print("Per-image details:", out)
