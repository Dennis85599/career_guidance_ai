from flask_login import UserMixin
from .extensions import login_manager, mysql


class Student(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email


class Admin(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email


@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, email FROM students WHERE id = %s", (user_id,))
    user = cur.fetchone()

    if user:
        return Student(user[0], user[1])
    return None
