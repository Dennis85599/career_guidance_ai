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
# KCSE CAREER REQUIREMENTS (Dataset-Derived)
# ===========================================

CAREER_REQUIREMENTS = {

# ================= HEALTH SCIENCES =================

"Doctor": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 10, "chemistry": 10, "history": 8}},
    "diploma": {"mean_grade": 5, "subjects": {"biology": 8, "chemistry": 8, "history": 6}}
},

"Nurse": {
    "degree": {"mean_grade": 7, "subjects": {"chemistry": 11, "biology": 10, "geography": 8}},
    "diploma": {"mean_grade": 5, "subjects": {"chemistry": 9, "biology": 8, "geography": 6}}
},

"Clinical Officer": {
    "degree": {"mean_grade": 7, "subjects": {"chemistry": 10, "biology": 10, "business": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"chemistry": 8, "biology": 8, "business": 5}}
},

"Pharmacist": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 11, "chemistry": 10, "history": 8}},
    "diploma": {"mean_grade": 5, "subjects": {"biology": 9, "chemistry": 8, "history": 6}}
},

"Lab Technologist": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 11, "chemistry": 10, "computer": 8}},
    "diploma": {"mean_grade": 5, "subjects": {"biology": 9, "chemistry": 8, "computer": 6}}
},

"Veterinary Officer": {
    "degree": {"mean_grade": 7, "subjects": {"geography": 11, "agriculture": 11, "biology": 10}},
    "diploma": {"mean_grade": 5, "subjects": {"geography": 9, "agriculture": 9, "biology": 8}}
},

# ================= ENGINEERING =================

"Civil Engineer": {
    "degree": {"mean_grade": 8, "subjects": {"physics": 11, "math": 11, "computer": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"physics": 9, "math": 9, "computer": 8}}
},

"Mechanical Engineer": {
    "degree": {"mean_grade": 8, "subjects": {"computer": 11, "math": 11, "physics": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"computer": 9, "math": 9, "physics": 8}}
},

"Electrical Engineer": {
    "degree": {"mean_grade": 8, "subjects": {"physics": 11, "math": 11, "computer": 11}},
    "diploma": {"mean_grade": 6, "subjects": {"physics": 9, "math": 9, "computer": 9}}
},

"Mechatronics Engineer": {
    "degree": {"mean_grade": 8, "subjects": {"math": 11, "computer": 10, "physics": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"math": 9, "computer": 8, "physics": 8}}
},

"Marine Engineer": {
    "degree": {"mean_grade": 6, "subjects": {"math": 7, "biology": 7, "agriculture": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"math": 5, "biology": 5, "agriculture": 5}}
},

"AI Engineer": {
    "degree": {"mean_grade": 8, "subjects": {"physics": 11, "math": 11, "computer": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"physics": 9, "math": 9, "computer": 8}}
},

# ================= ICT =================

"Software Developer": {
    "degree": {"mean_grade": 8, "subjects": {"math": 11, "physics": 11, "computer": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"math": 9, "physics": 9, "computer": 8}}
},

"Data Scientist": {
    "degree": {"mean_grade": 8, "subjects": {"math": 11, "computer": 11, "physics": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"math": 9, "computer": 9, "physics": 8}}
},

"Cybersecurity Analyst": {
    "degree": {"mean_grade": 8, "subjects": {"math": 11, "computer": 11, "physics": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"math": 9, "computer": 9, "physics": 8}}
},

# ================= LAW =================

"Lawyer": {
    "degree": {"mean_grade": 7, "subjects": {"english": 11, "history": 10, "business": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"english": 9, "history": 8, "business": 5}}
},

"Judge": {
    "degree": {"mean_grade": 7, "subjects": {"history": 11, "english": 10, "kiswahili": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"history": 9, "english": 8, "kiswahili": 5}}
},

"Public Administrator": {
    "degree": {"mean_grade": 7, "subjects": {"history": 11, "english": 11, "agriculture": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"history": 9, "english": 9, "agriculture": 5}}
},

# ================= BUSINESS =================

"Accountant": {
    "degree": {"mean_grade": 8, "subjects": {"business": 11, "math": 10, "geography": 8}},
    "diploma": {"mean_grade": 6, "subjects": {"business": 9, "math": 8, "geography": 6}}
},

"Actuary": {
    "degree": {"mean_grade": 7, "subjects": {"math": 11, "business": 10, "physics": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"math": 9, "business": 8, "physics": 5}}
},

"Economist": {
    "degree": {"mean_grade": 7, "subjects": {"business": 11, "math": 11, "biology": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"business": 9, "math": 9, "biology": 5}}
},

"Financial Analyst": {
    "degree": {"mean_grade": 7, "subjects": {"business": 11, "math": 10, "history": 8}},
    "diploma": {"mean_grade": 5, "subjects": {"business": 9, "math": 8, "history": 6}}
},

"Banker": {
    "degree": {"mean_grade": 7, "subjects": {"math": 10, "business": 10, "computer": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"math": 8, "business": 8, "computer": 5}}
},

# ================= EDUCATION =================

"Teacher": {
    "degree": {"mean_grade": 7, "subjects": {"english": 10, "computer": 7, "business": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"english": 8, "computer": 5, "business": 5}}
},

"Lecturer": {
    "degree": {"mean_grade": 7, "subjects": {"english": 11, "kiswahili": 8, "computer": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"english": 9, "kiswahili": 6, "computer": 5}}
},

"Education Officer": {
    "degree": {"mean_grade": 7, "subjects": {"english": 11, "physics": 8, "business": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"english": 9, "physics": 6, "business": 5}}
},

# ================= SECURITY =================

"Police Officer": {
    "degree": {"mean_grade": 6, "subjects": {"physics": 7, "geography": 7, "chemistry": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"physics": 5, "geography": 5, "chemistry": 5}}
},

"Military Officer": {
    "degree": {"mean_grade": 7, "subjects": {"physics": 7, "chemistry": 7, "geography": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"physics": 5, "chemistry": 5, "geography": 5}}
},

# ================= HOSPITALITY =================

"Chef": {
    "degree": {"mean_grade": 7, "subjects": {"physics": 7, "math": 7, "kiswahili": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"physics": 5, "math": 5, "kiswahili": 5}}
},

"Hotel Manager": {
    "degree": {"mean_grade": 7, "subjects": {"math": 8, "physics": 7, "kiswahili": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"math": 6, "physics": 5, "kiswahili": 5}}
},

"Tour Guide": {
    "degree": {"mean_grade": 6, "subjects": {"agriculture": 7, "physics": 7, "kiswahili": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"agriculture": 5, "physics": 5, "kiswahili": 5}}
},

# ================= CREATIVE =================

"Graphic Designer": {
    "degree": {"mean_grade": 6, "subjects": {"math": 8, "biology": 7, "kiswahili": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"math": 6, "biology": 5, "kiswahili": 5}}
},

"Animator": {
    "degree": {"mean_grade": 7, "subjects": {"english": 8, "agriculture": 7, "biology": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"english": 6, "agriculture": 5, "biology": 5}}
},

"Film Producer": {
    "degree": {"mean_grade": 6, "subjects": {"history": 7, "computer": 7, "physics": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"history": 5, "computer": 5, "physics": 5}}
},

"Journalist": {
    "degree": {"mean_grade": 7, "subjects": {"english": 7, "kiswahili": 7, "math": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"english": 5, "kiswahili": 5, "math": 5}}
},

# ================= AGRICULTURE =================

"Agronomist": {
    "degree": {"mean_grade": 8, "subjects": {"agriculture": 11, "geography": 10, "biology": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"agriculture": 9, "geography": 8, "biology": 8}}
},

"Forester": {
    "degree": {"mean_grade": 8, "subjects": {"biology": 11, "geography": 10, "agriculture": 10}},
    "diploma": {"mean_grade": 6, "subjects": {"biology": 9, "geography": 8, "agriculture": 8}}
},

"Environmental Scientist": {
    "degree": {"mean_grade": 7, "subjects": {"geography": 11, "biology": 11, "agriculture": 10}},
    "diploma": {"mean_grade": 5, "subjects": {"geography": 9, "biology": 9, "agriculture": 8}}
},

# ================= TRADES =================

"Electrician": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 7, "kiswahili": 7, "history": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"biology": 5, "kiswahili": 5, "history": 5}}
},

"Welder": {
    "degree": {"mean_grade": 6, "subjects": {"chemistry": 7, "computer": 7, "geography": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"chemistry": 5, "computer": 5, "geography": 5}}
},

"Automotive Technician": {
    "degree": {"mean_grade": 6, "subjects": {"business": 7, "physics": 7, "agriculture": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"business": 5, "physics": 5, "agriculture": 5}}
},

# ================= AVIATION =================

"Pilot": {
    "degree": {"mean_grade": 7, "subjects": {"biology": 7, "kiswahili": 7, "math": 7}},
    "diploma": {"mean_grade": 5, "subjects": {"biology": 5, "kiswahili": 5, "math": 5}}
},

"Logistics Officer": {
    "degree": {"mean_grade": 6, "subjects": {"kiswahili": 7, "math": 7, "computer": 7}},
    "diploma": {"mean_grade": 4, "subjects": {"kiswahili": 5, "math": 5, "computer": 5}}
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
