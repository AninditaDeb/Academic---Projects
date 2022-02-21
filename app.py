#Imports
from flask import Flask, redirect, url_for, render_template, request, session, flash
from flask_sqlalchemy import SQLAlchemy
import os

#Config
app = Flask(__name__)
app.secret_key = "hello"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

#Database
db = SQLAlchemy(app)

class users(db.Model):
    _id = db.Column("id", db.Integer, primary_key = True)
    name = db.Column("name", db.String(100))
    email = db.Column("email", db.String(100))

    def __init__(self, name, email):
        self.name = name
        self.email = email

#Home test view
@app.route("/")
def home():
   return render_template("index.html",content="Testing") 

#Database view
@app.route("/view")
def view():
    return render_template("view.html", values = users.query.all())

#Login
@app.route("/login",methods=["POST", "GET"])
def login():
    if request.method=="POST":
        user=request.form["nm"]
        session["user"]=user

        finding_user = users.query.filter_by(name = user).first()
        if finding_user:
            session["email"] = finding_user.email
        else:
            usr = users(user, "")
            db.session.add(usr)
            db.session.commit()

        flash(f"Login successful!", "info")   
        return(redirect(url_for("user")))
    else:
        if "user" in session:
            flash("Already logged in!")
            return redirect(url_for("user"))
        
        return render_template("login.html")

#User
@app.route("/user", methods = ["POST", "GET"])
def user():
    email = None
    if "user" in session:
        user = session["user"]

        if request.method == "POST":
            email = request.form["email"]
            session["email"] = email
            finding_user = users.query.filter_by(name = user).first()
            finding_user.email = email
            db.session.commit()
            flash("Email saved!")
        else:
            if "email" in session:
                email = session["email"]
                
        return render_template("user.html", email = email, user=user) 
    else:
        flash("You are  not logged in")
        return redirect(url_for("login"))

#Logout
@app.route("/logout")
def logout():
    if "user" in session:
        user=session["user"]
        flash(f"You have been logged out {user}", "info")         
    session.pop("user", None)
    session.pop("email", None)
    return redirect(url_for("login"))

#Main/ connection
if __name__ == '__main__':
    db.create_all()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port, debug = True)