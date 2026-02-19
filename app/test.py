import pandas as pd

df = pd.read_csv("data/cleaned_dataset.csv")

skill_cols = [
    'analytical_skill','numerical_skill','communication_skill',
    'creativity_skill','technical_skill','leadership_skill',
    'social_skill','physical_skill','artistic_skill',
    'entrepreneurial_skill'
]

print("\nSKILL VARIETY CHECK\n")
print(df.groupby("career_cluster")[skill_cols].mean())
