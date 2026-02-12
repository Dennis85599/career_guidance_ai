import joblib
import numpy as np

# =============================
# LOAD MODELS
# =============================
cluster_model = joblib.load("models/cluster_model.pkl")
career_models = joblib.load("models/career_models_by_cluster.pkl")

cluster_encoder = joblib.load("models/cluster_encoder.pkl")
career_encoder = joblib.load("models/career_encoder.pkl")
scaler = joblib.load("models/feature_scaler.pkl")

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
    # Aviation / Engineering
    "Pilot": {"math": 7, "physics": 7},
    "Aeronautical Engineer": {"math": 7, "physics": 7},
    "Civil Engineer": {"math": 6},
    "Electrical Engineer": {"math": 6, "physics": 6},
    "Mechanical Engineer": {"math": 6, "physics": 6},

    # Medical
    "Doctor": {"biology": 7, "chemistry": 7},
    "Pharmacist": {"chemistry": 6},

    # Flexible careers (no hard cutoffs)
    "Entrepreneur": {},
    "Tour Guide": {},
}

# =============================
# ELIGIBILITY CHECK
# =============================
def is_eligible(career, grades):
    """
    grades: dict {subject: score}
    """
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
    """
    student_data: full feature vector (subjects + skills)
    raw_grades: dict of actual student KCSE grades
    """

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

    # 2️⃣ Rank careers by probability
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

