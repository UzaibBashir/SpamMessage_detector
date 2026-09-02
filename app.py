from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/spam_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    prediction = model.predict([message])[0]

    return render_template(
        "index.html", prediction = prediction, message = message
    )

if __name__ == "main":
    app.run(debug=True)
    