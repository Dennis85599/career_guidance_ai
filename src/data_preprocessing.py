import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw_dataset.csv")

# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Validate KCSE subject ranges (1–12)
subject_cols = [
    'math','english','kiswahili','biology','chemistry','physics',
    'geography','history','business','computer','cre','agriculture'
]

for col in subject_cols:
    df = df[(df[col] >= 1) & (df[col] <= 12)]

# 3. Validate skill ranges (1–10)
skill_cols = [
    'analytical_skill','numerical_skill','communication_skill',
    'creativity_skill','technical_skill','leadership_skill',
    'social_skill','physical_skill','artistic_skill',
    'entrepreneurial_skill'
]

for col in skill_cols:
    df = df[(df[col] >= 1) & (df[col] <= 10)]

# 4. Drop unnecessary leakage column
# (mean_grade can bias prediction since it's derived from subjects)
df = df.drop(columns=['mean_grade'])

# Save cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

print("Data cleaning completed successfully!")
