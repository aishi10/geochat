import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

p = r"C:\Geochat\geochat\bigearthnet_test100\evaluation_details.csv"
df = pd.read_csv(p, keep_default_na=False)

classes = [
    "Arable land",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Pastures",
    "Inland waters",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Industrial or commercial units",
    "Urban fabric",
]

def parse_labels(s):
    return set(x.strip() for x in s.split("|") if x.strip())

true_sets = [parse_labels(x) for x in df["reference_labels"]]
pred_sets = [parse_labels(x) for x in df["predicted_labels"]]

y_true = [[int(c in labels) for c in classes] for labels in true_sets]
y_pred = [[int(c in labels) for c in classes] for labels in pred_sets]

p, r, f, support = precision_recall_fscore_support(
    y_true, y_pred, zero_division=0
)

result = pd.DataFrame({
    "Class": classes,
    "Precision": p.round(3),
    "Recall": r.round(3),
    "F1": f.round(3),
    "Reference_count": support,
    "Predicted_count": [
        sum(c in labels for labels in pred_sets) for c in classes
    ],
})

print(result.to_string(index=False))
result.to_csv(
    r"C:\Geochat\geochat\bigearthnet_test100\per_class_metrics.csv",
    index=False
)
print("\nSaved per_class_metrics.csv")
