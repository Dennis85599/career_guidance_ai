from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_login import login_user, login_required, logout_user, current_user
from .extensions import mysql, bcrypt
from .models import Student, Admin
from src.recommender import recommend_careers

main = Blueprint("main", __name__)

# =============================
# KCSE GRADE MAPPING
# =============================
GRADE_MAP = {
    "A":12, "A-":11, "B+":10, "B":9, "B-":8,
    "C+":7, "C":6, "C-":5, "D+":4, "D":3, "D-":2, "E":1
}

ALL_SUBJECTS = [
    "math","english","kiswahili","biology","chemistry","physics",
    "geography","history","business","computer","cre","agriculture"
]

# =============================
# HOME
# =============================
@main.route("/")
def home():
    return render_template("home.html")


# =============================
# SIGNUP
# =============================
@main.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO students (full_name, email, password_hash) VALUES (%s, %s, %s)",
            (name, email, password_hash)
        )
        mysql.connection.commit()

        return redirect(url_for("main.login"))

    return render_template("signup.html")


# =============================
# LOGIN
# =============================
@main.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute(
            "SELECT id, password_hash FROM students WHERE email = %s",
            (email,)
        )
        user = cur.fetchone()

        if user and bcrypt.check_password_hash(user[1], password):
            login_user(Student(user[0], email))
            return redirect(url_for("main.grades"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# =============================
# LOGOUT
# =============================
@main.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for("main.home"))


# =============================
# SUBJECTS
# =============================
@main.route("/subjects", methods=["GET", "POST"])
def subjects():
    if request.method == "POST":
        selected_subjects = request.form.getlist("subjects")

        if not selected_subjects:
            return render_template("subjects.html",
                                   error="Please select at least one subject.")

        session["subjects"] = selected_subjects
        return redirect(url_for("main.grades"))

    return render_template("subjects.html")


# =============================
# GRADES
# =============================
@main.route("/grades", methods=["GET", "POST"])
@login_required
def grades():
    if "subjects" not in session:
        return redirect(url_for("main.subjects"))

    if request.method == "POST":
        grades = {}

        for subject in session["subjects"]:
            letter = request.form.get(subject)
            grades[subject] = GRADE_MAP.get(letter, 1)

        session["grades"] = grades
        return redirect(url_for("main.skills"))

    return render_template("grades.html", subjects=session["subjects"])


# =============================
# SKILLS
# =============================
@main.route("/skills", methods=["GET", "POST"])
@login_required
def skills():
    if "grades" not in session:
        return redirect(url_for("main.subjects"))

    if request.method == "POST":
        SKILL_ORDER = [
            "analytical","numerical","communication","creativity","technical",
            "leadership","social","practical","artistic","entrepreneurial"
        ]

        skills = [int(request.form.get(f"skill{i}")) for i in range(10)]

        raw_grades = session["grades"]
        final_subjects = [raw_grades.get(sub, 1) for sub in ALL_SUBJECTS]

        student_data = final_subjects + skills

        cluster, careers = recommend_careers(
            student_data=student_data,
            raw_grades=raw_grades
        )

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO recommendations (student_id, cluster) VALUES (%s, %s)",
            (current_user.id, cluster)
        )
        mysql.connection.commit()

        return render_template("result.html",
                               cluster=cluster,
                               careers=careers)

    return render_template("skills.html")


# =============================
# HELP
# =============================
@main.route("/help")
def help():
    return render_template("help.html")
