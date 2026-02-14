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
# LOAD MODELS
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
# KCSE CAREER REQUIREMENTS
# =============================
CAREER_REQUIREMENTS = {

# ================= HEALTH SCIENCES =================

"Doctor": {
    "degree": {"mean_grade": 8, "subjects": {"biology": 8, "chemistry": 8, "physics": 6}},
    "diploma": {"mean_grade": 6, "subjects": {"biology": 6, "chemistry": 6}}
},

"Nurse": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 7, "chemistry": 6}},
    "diploma": {"mean_grade": 5, "subjects": {"biology": 5}}
},

"Clinical Officer": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"biology": 5}}
},

"Pharmacist": {
    "degree": {"mean_grade": 8, "subjects": {"chemistry": 8, "biology": 7}},
    "diploma": {"mean_grade": 6, "subjects": {"chemistry": 6}}
},

"Lab Technologist": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 6, "chemistry": 6}},
    "diploma": {"mean_grade": 5}
},

"Veterinary Officer": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 7, "chemistry": 6}},
    "diploma": {"mean_grade": 5}
},

# ================= ENGINEERING =================

"Civil Engineer": {
    "degree": {"mean_grade": 7, "subjects": {"math": 7, "physics": 6}},
    "diploma": {"mean_grade": 5, "subjects": {"math": 5}}
},

"Mechanical Engineer": {
    "degree": {"mean_grade": 7, "subjects": {"math": 7, "physics": 6}},
    "diploma": {"mean_grade": 5}
},

"Electrical Engineer": {
    "degree": {"mean_grade": 7, "subjects": {"math": 7, "physics": 6}},
    "diploma": {"mean_grade": 5}
},

"Mechatronics Engineer": {
    "degree": {"mean_grade": 8, "subjects": {"math": 8, "physics": 7}},
    "diploma": {"mean_grade": 6}
},

"Marine Engineer": {
    "degree": {"mean_grade": 7, "subjects": {"math": 7, "physics": 6}},
    "diploma": {"mean_grade": 5}
},

"AI Engineer": {
    "degree": {"mean_grade": 8, "subjects": {"math": 8}},
    "diploma": {"mean_grade": 6}
},

# ================= ICT =================

"Software Developer": {
    "degree": {"mean_grade": 7, "subjects": {"math": 6}},
    "diploma": {"mean_grade": 5}
},

"Data Scientist": {
    "degree": {"mean_grade": 8, "subjects": {"math": 8}},
    "diploma": {"mean_grade": 6}
},

"Cybersecurity Analyst": {
    "degree": {"mean_grade": 7, "subjects": {"math": 6}},
    "diploma": {"mean_grade": 5}
},

# ================= LAW =================

"Lawyer": {
    "degree": {"mean_grade": 8, "subjects": {"english": 7, "kiswahili": 6}},
    "diploma": None
},

"Judge": {
    "degree": {"mean_grade": 9, "subjects": {"english": 8}},
    "diploma": None
},

"Public Administrator": {
    "degree": {"mean_grade": 7},
    "diploma": {"mean_grade": 5}
},

# ================= BUSINESS =================

"Accountant": {
    "degree": {"mean_grade": 7, "subjects": {"math": 6}},
    "diploma": {"mean_grade": 5}
},

"Actuary": {
    "degree": {"mean_grade": 9, "subjects": {"math": 9}},
    "diploma": None
},

"Economist": {
    "degree": {"mean_grade": 8, "subjects": {"math": 7}},
    "diploma": {"mean_grade": 6}
},

"Financial Analyst": {
    "degree": {"mean_grade": 7, "subjects": {"math": 6}},
    "diploma": {"mean_grade": 5}
},

"Banker": {
    "degree": {"mean_grade": 7},
    "diploma": {"mean_grade": 5}
},

# ================= EDUCATION =================

"Teacher": {
    "degree": {"mean_grade": 7},
    "diploma": {"mean_grade": 5}
},

"Lecturer": {
    "degree": {"mean_grade": 8},
    "diploma": None
},

"Education Officer": {
    "degree": {"mean_grade": 7},
    "diploma": {"mean_grade": 5}
},

# ================= SECURITY =================

"Police Officer": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Military Officer": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

# ================= HOSPITALITY =================

"Chef": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Hotel Manager": {
    "degree": {"mean_grade": 7},
    "diploma": {"mean_grade": 5}
},

"Tour Guide": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

# ================= CREATIVE =================

"Graphic Designer": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Animator": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Film Producer": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Journalist": {
    "degree": {"mean_grade": 7, "subjects": {"english": 6}},
    "diploma": {"mean_grade": 5}
},

# ================= AGRICULTURE =================

"Agronomist": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 6}},
    "diploma": {"mean_grade": 5}
},

"Forester": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Environmental Scientist": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 6}},
    "diploma": {"mean_grade": 5}
},

# ================= TRADES =================

"Electrician": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Welder": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

"Automotive Technician": {
    "degree": {"mean_grade": 6},
    "diploma": {"mean_grade": 4}
},

# ================= AVIATION =================

"Pilot": {
    "degree": {"mean_grade": 8, "subjects": {"math": 8, "physics": 7}},
    "diploma": {"mean_grade": 6}
},

"Logistics Officer": {
    "degree": {"mean_grade": 7},
    "diploma": {"mean_grade": 5}
}
}

# =============================
# SAFE MEAN GRADE
# =============================
# =============================
# SAFE MEAN GRADE
# =============================
def calculate_mean_grade(grades):
    if not grades:
        return 0.0
    return round(sum(grades.values()) / max(len(grades), 1), 2)


# =============================
# GENERIC PATHWAY CHECK
# =============================
def evaluate_pathway(pathway_rules, grades):

    if not pathway_rules:
        return {"eligible": False, "missing_subjects": []}

    mean_required = pathway_rules.get("mean_grade", 0)
    subject_rules = pathway_rules.get("subjects", {})

    student_mean = calculate_mean_grade(grades)

    if student_mean < mean_required:
        return {
            "eligible": False,
            "missing_subjects": ["Mean grade below requirement"]
        }

    missing = []

    for subject, min_score in subject_rules.items():
        if grades.get(subject, 0) < min_score:
            missing.append(subject)

    return {
        "eligible": len(missing) == 0,
        "missing_subjects": missing
    }


# =============================
# 4-LEVEL CAREER EVALUATION
# =============================
def evaluate_career_kcse(career, grades):

    rules = CAREER_REQUIREMENTS.get(career, {})

    # -------- Degree --------
    degree_eval = evaluate_pathway(rules.get("degree"), grades)
    if degree_eval["eligible"]:
        return "Degree", "Eligible for direct university admission."

    # -------- Diploma --------
    diploma_eval = evaluate_pathway(rules.get("diploma"), grades)
    if diploma_eval["eligible"]:
        return "Diploma", "Eligible for diploma program. Can upgrade to degree later."

    # -------- Certificate (Auto if not defined) --------
    certificate_rules = rules.get("certificate") or {
        "mean_grade": 3  # KCSE D
    }

    certificate_eval = evaluate_pathway(certificate_rules, grades)
    if certificate_eval["eligible"]:
        return "Certificate", "Eligible for certificate program."

    # -------- Artisan (Auto if not defined) --------
    artisan_rules = rules.get("artisan") or {
        "mean_grade": 2  # KCSE D-
    }

    artisan_eval = evaluate_pathway(artisan_rules, grades)
    if artisan_eval["eligible"]:
        return "Artisan", "Eligible for artisan/TVET training."

    # -------- Not Eligible --------
    return "Not Eligible", "Minimum KCSE requirements not met."

# =============================
# MAIN RECOMMENDER
# =============================
def recommend_careers(student_data, raw_grades, top_k=3):

    try:
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

        # 3️⃣ Apply KCSE pathway logic
        final_careers = []

        for career in ranked_careers:

            pathway, advice = evaluate_career_kcse(career, raw_grades)

            final_careers.append({
                "career": career,
                "pathway": pathway,
                "advice": advice
            })

            if len(final_careers) == top_k:
                break

        return cluster_name, final_careers

    except Exception as e:
        print("❌ Recommendation Error:", str(e))
        return "Unknown Cluster", []
