from flask import Flask
from .extensions import mysql, bcrypt, login_manager


def create_app():
    app = Flask(__name__)

    # Secret key
    app.secret_key = "career_guidance_secret"

    # =============================
    # DATABASE CONFIG
    # =============================
    app.config["MYSQL_HOST"] = "localhost"
    app.config["MYSQL_USER"] = "root"
    app.config["MYSQL_PASSWORD"] = ""
    app.config["MYSQL_DB"] = "career_guidance_db"

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
