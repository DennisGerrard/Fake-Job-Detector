#Stage 2 test: load the dataset and describe it."""

from pathlib import Path
import pandas as pd

DATA = Path("data")
EXPECTED = [
    "title", "location", "salary_range", "company_profile", "description",
    "requirements", "benefits", "telecommuting", "has_company_logo",
    "has_questions", "employment_type", "required_experience",
    "required_education", "fraudulent",
]


def find_dataset():
    for name in ("fake_job_postings.csv", "sample_job_postings.csv"):
        if (DATA / name).exists():
            return DATA / name
    raise SystemExit("No CSV in data/. Download the Kaggle file or run make_sample_data.py")


path = find_dataset()
df = pd.read_csv(path)
print(f"File     : {path.name}")
print(f"Shape    : {df.shape[0]:,} rows x {df.shape[1]} columns\n")

missing_cols = [c for c in EXPECTED if c not in df.columns]
print("Required columns:", "all present" if not missing_cols else f"MISSING {missing_cols}")

print("\nClass balance")
counts = df["fraudulent"].value_counts().sort_index()
for label, count in counts.items():
    name = "genuine" if label == 0 else "fake"
    print(f"  {name:<8} {count:>7,}  ({count / len(df) * 100:5.2f}%)")
print(f"  Predicting 'genuine' for everything would already score "
      f"{counts.get(0, 0) / len(df) * 100:.2f}% accuracy.")

print("\nEmpty values per column")
for col in EXPECTED:
    if col in df.columns:
        blank = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        print(f"  {col:<22} {blank:>7,}  ({blank / len(df) * 100:5.1f}%)")

print("\nHow the two classes differ")
df["desc_len"] = df["description"].fillna("").astype(str).str.len()
df["profile_len"] = df["company_profile"].fillna("").astype(str).str.len()
summary = df.groupby("fraudulent")[
    ["desc_len", "profile_len", "has_company_logo", "has_questions", "telecommuting"]
].mean().round(2)
summary.index = ["genuine", "fake"]
print(summary.to_string())

print("\nOne fraudulent example")
sample = df[df.fraudulent == 1].iloc[0]
print(f"  title: {sample['title']}")
print(f"  body : {str(sample['description'])[:200]}...")

print("\nStage 2 passed." if not missing_cols else "\nStage 2 failed — column names don't match.")