import pandas as pd

ASPECTS = ["product", "price", "service", "variety"]
LABELS = ["neg", "neu", "pos"]

df = pd.read_csv("data/absa_mobile_shop.csv")

print("Total rows:", len(df))
print("\n=== Label Distribution (per aspect) ===")
for a in ASPECTS:
    print(f"\n[{a}]")
    counts = df[a].astype(str).str.strip().str.lower().value_counts()
    for lab in LABELS:
        print(f"  {lab}: {int(counts.get(lab, 0))}")