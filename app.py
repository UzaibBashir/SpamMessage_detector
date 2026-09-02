from flask import Flask, render_template, request
import joblib
import string
from nltk.corpus import stopwords
import os

# Define tokenisation function in __main__ scope for pickle to find it
def tokenisation(mess):
    """
    Process text by removing punctuation and stopwords.
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Removing stopwords and returning the list
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'app', 'templates'))

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

if __name__ == "__main__":
    app.run(debug=True)
    