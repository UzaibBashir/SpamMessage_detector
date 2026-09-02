import joblib
import string
from nltk.corpus import stopwords

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


model = joblib.load("model/spam_pipeline.pkl")

messages = ["Congratulations! you have won a free prize. Call now!", "Hey, are we meeting today?"]
preds = model.predict(messages)

for message, prediction in zip(messages, preds):
    print(message, "\t", prediction)
    print()