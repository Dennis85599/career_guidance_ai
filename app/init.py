from flask import Flask
from .extensions import mysql, bcrypt, login_manager
import os


def create_app():
    app = Flask(__name__)

    # Secret key
    app.secret_key = os.getenv("SECRET_KEY", "career_guidance_secret")

    # =============================
    # DATABASE CONFIG
    # =============================
    app.config["MYSQL_HOST"] = os.getenv("MYSQL_HOST")
    app.config["MYSQL_USER"] = os.getenv("MYSQL_USER")
    app.config["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD")
    app.config["MYSQL_DB"] = os.getenv("MYSQL_DB")
    app.config["MYSQL_PORT"] = int(os.getenv("MYSQL_PORT", 3306))

    app.config["MYSQL_SSL"] = {"ssl": {}}

    # =============================
    # INITIALIZE EXTENSIONS
    # =============================
    mysql.init_app(app)
    bcrypt.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = "main.login"

    # =============================
    # REGISTER BLUEPRINTS
    # =============================
    from .routes import main
    app.register_blueprint(main)

    return app