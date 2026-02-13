import joblib
import numpy as np
import os
import requests

# =============================
# MODEL DIRECTORY
# =============================
MODEL_DIR = "models"

# =============================
# GOOGLE DRIVE FILE IDS
# =============================
MODEL_FILES = {
    "career_encoder.pkl": "1q1an6T4lI-J4Q3HHCzqp5d8yJLQJhljk",
    "cluster_encoder.pkl": "1yr7v15kSsopcmCfoifumeDrrZ12dIQ6d",
    "cluster_model.pkl": "1c9kKWJRD4XxQA1yBJoi7AIxiUWyk_0fG",
    "career_model.pkl": "1Iycpr42Hc9B7gSA9uVBVyVCiHERGhp6O",
    "feature_scaler.pkl": "1kL7PRu5jxJvCTicoGNeiOuPOodTOdc1S",
    "career_models_by_cluster.pkl": "1qwDmtY1pMpygbaOMVBwuOTljg-aR_KIv",
}

# =============================
# DOWNLOAD FUNCTION
# =============================
def download_file(file_id, destination):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded {destination}")
    else:
        raise Exception(f"❌ Failed to download {destination}")

# =============================
# ENSURE MODELS EXIST
# =============================
def ensure_models_exist():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for filename, file_id in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"⬇ Downloading {filename}...")
            download_file(file_id, path)
        else:
            print(f"✔ {filename} already exists")

# =============================
# LOAD MODELS ONCE (CRITICAL FIX)
# =============================
ensure_models_exist()

print("🔄 Loading ML models...")

cluster_model = joblib.load(os.path.join(MODEL_DIR, "cluster_model.pkl"))
career_models = joblib.load(os.path.join(MODEL_DIR, "career_models_by_cluster.pkl"))
cluster_encoder = joblib.load(os.path.join(MODEL_DIR, "cluster_encoder.pkl"))
career_encoder = joblib.load(os.path.join(MODEL_DIR, "career_encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))

print("✅ All models loaded successfully")

# =============================
# FEATURE ORDER (MUST MATCH TRAINING)
# =============================
SUBJECT_COLS = [
    'math','english','kiswahili','biology','chemistry','physics',
    'geography','history','business','computer','cre','agriculture'
]

SKILL_COLS = [
    'analytical_skill','numerical_skill','communication_skill',
    'creativity_skill','technical_skill','leadership_skill',
    'social_skill','physical_skill','artistic_skill',
    'entrepreneurial_skill'
]

FEATURE_COUNT = len(SUBJECT_COLS) + len(SKILL_COLS)

# =============================
# CAREER ELIGIBILITY RULES
# =============================
CAREER_REQUIREMENTS = {
    "Pilot": {"math": 7, "physics": 7},
    "Aeronautical Engineer": {"math": 7, "physics": 7},
    "Civil Engineer": {"math": 6},
    "Electrical Engineer": {"math": 6, "physics": 6},
    "Mechanical Engineer": {"math": 6, "physics": 6},
    "Doctor": {"biology": 7, "chemistry": 7},
    "Pharmacist": {"chemistry": 6},
    "Entrepreneur": {},
    "Tour Guide": {},
}

# =============================
# ELIGIBILITY CHECK
# =============================
def is_eligible(career, grades):
    if career not in CAREER_REQUIREMENTS:
        return True

    rules = CAREER_REQUIREMENTS[career]

    for subject, min_score in rules.items():
        if grades.get(subject, 0) < min_score:
            return False

    return True

# =============================
# MAIN RECOMMENDER
# =============================
def recommend_careers(student_data, raw_grades, top_k=3):

    if len(student_data) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {len(student_data)}")

    # Scale input
    student_array = np.array(student_data).reshape(1, -1)
    student_scaled = scaler.transform(student_array)

    # 1️⃣ Predict cluster
    cluster_id = cluster_model.predict(student_scaled)[0]
    cluster_name = cluster_encoder.inverse_transform([cluster_id])[0]

    if cluster_id not in career_models:
        return cluster_name, []

    career_model = career_models[cluster_id]

    # 2️⃣ Rank careers
    probs = career_model.predict_proba(student_scaled)[0]
    classes = career_model.classes_

    ranked_indices = np.argsort(probs)[::-1]
    ranked_ids = classes[ranked_indices]
    ranked_careers = career_encoder.inverse_transform(ranked_ids)

    # 3️⃣ Apply eligibility rules
    final_careers = []

    for career in ranked_careers:
        if is_eligible(career, raw_grades):
            final_careers.append(career)

        if len(final_careers) == top_k:
            break

    return cluster_name, final_careers
