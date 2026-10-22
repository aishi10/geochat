import pandas as pd

p = r"C:\Geochat\geochat\bigearthnet_test100\evaluation_details.csv"
df = pd.read_csv(p, keep_default_na=False)

for i, row in df.head(10).iterrows():
    print("\n" + "=" * 70)
    print("Image:", i)
    print("REFERENCE:", row["reference_labels"])
    print("DETECTED: ", row["predicted_labels"])
    print("RAW ANSWER:", row["raw_answer"])
