import pymysql
pymysql.install_as_MySQLdb()

from flask import Flask
from .extensions import mysql, bcrypt, login_manager
from .model_loader import ensure_models_exist
import os


def create_app():
    app = Flask(__name__)

    app.secret_key = os.environ.get("SECRET_KEY", "devkey")

    app.config["MYSQL_HOST"] = os.environ.get("MYSQL_HOST", "localhost")
    app.config["MYSQL_USER"] = os.environ.get("MYSQL_USER", "root")
    app.config["MYSQL_PASSWORD"] = os.environ.get("MYSQL_PASSWORD", "")
    app.config["MYSQL_DB"] = os.environ.get("MYSQL_DB", "career_guidance_db")

    mysql.init_app(app)
    bcrypt.init_app(app)

    login_manager.init_app(app)
    login_manager.login_view = "main.login"

    # ✅ Download models when app starts
    ensure_models_exist()

    from .routes import main
    app.register_blueprint(main)

    return app
