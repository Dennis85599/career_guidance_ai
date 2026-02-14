import joblib
import numpy as np
import os
import requests

# =============================
# MODEL DIRECTORY
# =============================
MODEL_DIR = "models"

MODEL_FILES = {
    "career_encoder.pkl": "1q1an6T4lI-J4Q3HHCzqp5d8yJLQJhljk",
    "cluster_encoder.pkl": "1yr7v15kSsopcmCfoifumeDrrZ12dIQ6d",
    "cluster_model.pkl": "1c9kKWJRD4XxQA1yBJoi7AIxiUWyk_0fG",
    "feature_scaler.pkl": "1kL7PRu5jxJvCTicoGNeiOuPOodTOdc1S",
    "career_models_by_cluster.pkl": "1qwDmtY1pMpygbaOMVBwuOTljg-aR_KIv",
}

# =============================
# DOWNLOAD MODELS
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

def ensure_models_exist():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for filename, file_id in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"⬇ Downloading {filename}...")
            download_file(file_id, path)
        else:
            print(f"✔ {filename} exists")

# =============================
# LOAD MODELS ONCE
# =============================
ensure_models_exist()

print("🔄 Loading ML models...")

cluster_model = joblib.load(os.path.join(MODEL_DIR, "cluster_model.pkl"))
career_models = joblib.load(os.path.join(MODEL_DIR, "career_models_by_cluster.pkl"))
cluster_encoder = joblib.load(os.path.join(MODEL_DIR, "cluster_encoder.pkl"))
career_encoder = joblib.load(os.path.join(MODEL_DIR, "career_encoder.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))

print("✅ Models loaded successfully")

# =============================
# FEATURE ORDER
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

FEATURE_NAMES = SUBJECT_COLS + SKILL_COLS
FEATURE_COUNT = len(FEATURE_NAMES)

# =============================
# KCSE REQUIREMENTS
# =============================
CAREER_REQUIREMENTS = {
    "Doctor": {
        "degree": {
            "mean_grade": 7.0,
            "subjects": {"biology": 7, "chemistry": 7, "english": 6}
        },
        "diploma": {
            "mean_grade": 6.0,
            "subjects": {"biology": 6, "chemistry": 6}
        }
    },
    "Pharmacist": {
        "degree": {
            "mean_grade": 7.0,
            "subjects": {"chemistry": 7, "biology": 7, "math": 6}
        },
        "diploma": {
            "mean_grade": 6.0,
            "subjects": {"chemistry": 6, "biology": 6}
        }
    },
    "Software Engineer": {
        "degree": {
            "mean_grade": 6.0,
            "subjects": {"math": 6, "physics": 6}
        },
        "diploma": {
            "mean_grade": 5.0,
            "subjects": {"math": 5}
        }
    }
}

# =============================
# KCSE LOGIC
# =============================
def calculate_mean_grade(grades):
    if not grades:
        return 0.0
    return round(sum(grades.values()) / len(grades), 2)

def evaluate_pathway(rules, grades):
    if not rules:
        return False

    if calculate_mean_grade(grades) < rules.get("mean_grade", 0):
        return False

    for subject, min_score in rules.get("subjects", {}).items():
        if grades.get(subject, 0) < min_score:
            return False

    return True

def evaluate_career_kcse(career, grades):

    if career not in CAREER_REQUIREMENTS:
        return "Degree", "General university eligibility applies."

    rules = CAREER_REQUIREMENTS[career]

    if evaluate_pathway(rules.get("degree"), grades):
        return "Degree", "Eligible for direct university entry."

    if evaluate_pathway(rules.get("diploma"), grades):
        return "Diploma", "Consider diploma pathway then upgrade."

    return "Not Eligible", "Minimum KCSE requirements not met."

# =============================
# LIGHTWEIGHT EXPLAINABLE AI
# =============================
def generate_explanation(cluster_id):

    model = career_models.get(cluster_id)

    if not model:
        return "AI explanation unavailable."

    try:
        importances = model.feature_importances_

        feature_impact = list(zip(FEATURE_NAMES, importances))
        feature_impact.sort(key=lambda x: x[1], reverse=True)

        top_features = feature_impact[:4]

        important = [
            f.replace("_", " ").title()
            for f, _ in top_features
        ]

        return "Strong influence from: " + ", ".join(important)

    except Exception as e:
        print("Explanation error:", e)
        return "AI reasoning unavailable."

# =============================
# MAIN RECOMMENDER
# =============================
def recommend_careers(student_data, raw_grades, top_k=3):

    try:
        if len(student_data) != FEATURE_COUNT:
            raise ValueError("Feature count mismatch.")

        student_array = np.array(student_data).reshape(1, -1)
        student_scaled = scaler.transform(student_array)

        # Predict cluster
        cluster_id = cluster_model.predict(student_scaled)[0]
        cluster_name = cluster_encoder.inverse_transform([cluster_id])[0]

        if cluster_id not in career_models:
            return cluster_name, []

        career_model = career_models[cluster_id]

        probs = career_model.predict_proba(student_scaled)[0]
        classes = career_model.classes_

        ranked_indices = np.argsort(probs)[::-1]

        final_careers = []

        for idx in ranked_indices[:top_k]:

            career_id = classes[idx]
            career = career_encoder.inverse_transform([career_id])[0]
            confidence = round(float(probs[idx] * 100), 2)

            pathway, advice = evaluate_career_kcse(career, raw_grades)

            explanation = generate_explanation(cluster_id)

            final_careers.append({
                "career": career,
                "pathway": pathway,
                "advice": advice,
                "confidence": confidence,
                "explanation": explanation
            })

        return cluster_name, final_careers

    except Exception as e:
        print("❌ Recommendation Error:", str(e))
        return "Unknown Cluster", []
