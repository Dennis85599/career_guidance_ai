import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_dataset.csv")

subject_cols = [
    'math','english','kiswahili','biology','chemistry','physics',
    'geography','history','business','computer','cre','agriculture'
]

skill_cols = [
    'analytical_skill','numerical_skill','communication_skill',
    'creativity_skill','technical_skill','leadership_skill',
    'social_skill','physical_skill','artistic_skill',
    'entrepreneurial_skill'
]

X = df[subject_cols + skill_cols]

# Encode career cluster
cluster_encoder = LabelEncoder()
y_cluster = cluster_encoder.fit_transform(df['career_cluster'])

# Encode career
career_encoder = LabelEncoder()
y_career = career_encoder.fit_transform(df['recommended_career'])

# Save encoders (VERY IMPORTANT for deployment)
joblib.dump(cluster_encoder, "models/cluster_encoder.pkl")
joblib.dump(career_encoder, "models/career_encoder.pkl")

print("Feature engineering completed successfully!")
print(f"Number of career clusters: {len(cluster_encoder.classes_)}")
print(f"Number of careers: {len(career_encoder.classes_)}")
