import joblib
import string
from nltk.corpus import stopwords



model = joblib.load("model/spam_pipeline.pkl")

messages = ["Congratulations! you have won a free prize. Call now!", "Hey, are we meeting today?"]
preds = model.predict(messages)

for message, prediction in zip(messages, preds):
    print(message, "\t", prediction)
    print()