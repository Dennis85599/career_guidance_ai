import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/cleaned_dataset.csv")

# Feature columns
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

# -------------------------
# ENCODE TARGETS
# -------------------------
cluster_encoder = LabelEncoder()
career_encoder = LabelEncoder()

y_cluster = cluster_encoder.fit_transform(df['career_cluster'])
y_career = career_encoder.fit_transform(df['recommended_career'])

# -------------------------
# CORRECT SINGLE SPLIT
# -------------------------
X_train, X_test, y_cluster_train, y_cluster_test, y_career_train, y_career_test = train_test_split(
    X,
    y_cluster,
    y_career,
    test_size=0.2,
    random_state=42,
    stratify=y_cluster
)

# -------------------------
# CLUSTER MODEL
# -------------------------
cluster_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cluster_model.fit(X_train, y_cluster_train)
cluster_preds = cluster_model.predict(X_test)
cluster_accuracy = accuracy_score(y_cluster_test, cluster_preds)

# -------------------------
# CAREER MODEL (BEST FLAT RF)
# -------------------------
career_model = RandomForestClassifier(
    n_estimators=600,
    max_depth=18,
    min_samples_split=6,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

career_model.fit(X_train, y_career_train)
career_preds = career_model.predict(X_test)
career_accuracy = accuracy_score(y_career_test, career_preds)

# -------------------------
# TOP-3 ACCURACY (VERY IMPORTANT)
# -------------------------
career_probs = career_model.predict_proba(X_test)
top3_preds = np.argsort(career_probs, axis=1)[:, -3:]

top3_accuracy = np.mean([
    y_career_test[i] in top3_preds[i]
    for i in range(len(y_career_test))
])

# -------------------------
# SAVE MODELS
# -------------------------
joblib.dump(cluster_model, "models/cluster_model.pkl")
joblib.dump(career_model, "models/career_model.pkl")
joblib.dump(cluster_encoder, "models/cluster_encoder.pkl")
joblib.dump(career_encoder, "models/career_encoder.pkl")

# -------------------------
# RESULTS
# -------------------------
print("\n✅ MODEL TRAINING COMPLETED\n")
print(f"🎯 Career Cluster Accuracy : {cluster_accuracy:.2f}")
print(f"🎓 Career Top-1 Accuracy   : {career_accuracy:.2f}")
print(f"🥇 Career Top-3 Accuracy   : {top3_accuracy:.2f}")
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# =============================
# LOAD DATA
# =============================
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

feature_cols = subject_cols + skill_cols
X = df[feature_cols]

# =============================
# ENCODE TARGETS
# =============================
cluster_encoder = LabelEncoder()
career_encoder = LabelEncoder()

df['cluster_enc'] = cluster_encoder.fit_transform(df['career_cluster'])
df['career_enc'] = career_encoder.fit_transform(df['recommended_career'])

# =============================
# SCALE FEATURES
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================
# TRAIN CLUSTER MODEL
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df['cluster_enc'],
    test_size=0.2,
    random_state=42,
    stratify=df['cluster_enc']
)

cluster_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cluster_model.fit(X_train, y_train)
cluster_preds = cluster_model.predict(X_test)
cluster_acc = accuracy_score(y_test, cluster_preds)

print("\n✅ MODEL TRAINING COMPLETED")
print(f"\n🎯 Career Cluster Accuracy : {cluster_acc:.2f}")

# =============================
# TRAIN CAREER MODELS PER CLUSTER
# =============================
career_models = {}

print("\n🎓 Training Career Models per Cluster...\n")

top1_scores = []
top3_scores = []

for cluster_label in df['career_cluster'].unique():
    cluster_id = cluster_encoder.transform([cluster_label])[0]

    cluster_mask = df['cluster_enc'] == cluster_id
    X_cluster = X_scaled[cluster_mask]
    y_cluster = df.loc[cluster_mask, 'career_enc']

    # Skip tiny clusters
    if y_cluster.nunique() < 3:
        print(f"   {cluster_label:<30} → Skipped (too few careers)")
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_cluster,
        y_cluster,
        test_size=0.2,
        random_state=42,
        stratify=y_cluster
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)

    # ---------- TOP-K EVALUATION (FIXED) ----------
    probs = model.predict_proba(X_te)
    classes = model.classes_

    top1_preds = classes[np.argmax(probs, axis=1)]
    top3_preds = classes[np.argsort(probs, axis=1)[:, -3:]]

    top1_acc = accuracy_score(y_te, top1_preds)
    top3_acc = np.mean([
        y_te.iloc[i] in top3_preds[i]
        for i in range(len(y_te))
    ])

    career_models[cluster_id] = model

    top1_scores.append(top1_acc)
    top3_scores.append(top3_acc)

    print(
        f"   {cluster_label:<30} → "
        f"Top-1: {top1_acc:.2f} | Top-3: {top3_acc:.2f}"
    )

# =============================
# OVERALL CAREER PERFORMANCE
# =============================
print("\n📊 OVERALL CAREER PERFORMANCE")
print(f"🎓 Career Top-1 Accuracy   : {np.mean(top1_scores):.2f}")
print(f"🥇 Career Top-3 Accuracy   : {np.mean(top3_scores):.2f}")

# =============================
# SAVE MODELS
# =============================
joblib.dump(cluster_model, "models/cluster_model.pkl")
joblib.dump(career_models, "models/career_models_by_cluster.pkl")
joblib.dump(cluster_encoder, "models/cluster_encoder.pkl")
joblib.dump(career_encoder, "models/career_encoder.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl")

print("\n💾 Models saved successfully.")
