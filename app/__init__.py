import os
import pymysql

pymysql.install_as_MySQLdb()

from flask import Flask
from .extensions import mysql, bcrypt, login_manager
from .model_loader import ensure_models_exist


def create_app():
    app = Flask(__name__)

    # =============================
    # SECRET KEY
    # =============================
    app.secret_key = os.environ.get("SECRET_KEY", "devkey")

    # =============================
    # MYSQL CONFIG (Railway Ready)
    # =============================
    app.config["MYSQL_HOST"] = os.environ.get("MYSQL_HOST")
    app.config["MYSQL_USER"] = os.environ.get("MYSQL_USER")
    app.config["MYSQL_PASSWORD"] = os.environ.get("MYSQL_PASSWORD")
    app.config["MYSQL_DB"] = os.environ.get("MYSQL_DB")
    app.config["MYSQL_PORT"] = int(os.environ.get("MYSQL_PORT", 3306))

    # 🔥 Required for Railway external MySQL
    app.config["MYSQL_SSL"] = {"ssl": {}}

    # =============================
    # INITIALIZE EXTENSIONS
    # =============================
    mysql.init_app(app)
    bcrypt.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = "main.login"

    # =============================
    # ENSURE MODELS EXIST (Render)
    # =============================
    ensure_models_exist()

    # =============================
    # REGISTER BLUEPRINTS
    # =============================
    from .routes import main
    app.register_blueprint(main)

    return app
