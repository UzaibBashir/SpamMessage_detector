# SMS Spam Detector: Interview Guide

## 30-second project explanation

This project is an SMS spam classification web application. It uses a
scikit-learn machine learning pipeline trained on the UCI SMS Spam Collection
dataset. The pipeline removes punctuation and English stopwords, converts the
message into TF-IDF features, and uses a Multinomial Naive Bayes classifier to
predict whether the message is spam or ham. The trained pipeline is serialized
with Joblib and served through a Flask web application.

## How to explain the architecture

The system has an offline training stage and a runtime prediction stage:

1. The SMS dataset is explored and split into training and testing data in a
   notebook.
2. Text preprocessing tokenizes each message, removes punctuation and
   stopwords, and prepares the words for feature extraction.
3. `CountVectorizer` converts the text into token counts.
4. `TfidfTransformer` gives more importance to informative words and less
   importance to words that appear frequently across many messages.
5. `MultinomialNB` learns the relationship between the text features and the
   `spam` or `ham` labels.
6. The complete pipeline is saved as `model/spam_pipeline.pkl`.
7. At runtime, `app.py` loads the saved pipeline once when the Flask server
   starts.
8. A user submits a message through `app/templates/index.html`. Flask receives
   it at `POST /predict`, sends it to the pipeline, and renders the prediction
   on the page.

There is also a command-line path in `model.py` that loads the same saved
pipeline and prints predictions for sample messages.

## Why these technologies?

- **Flask**: Lightweight framework for exposing the prediction form and HTTP
  endpoint.
- **scikit-learn**: Provides the vectorization, TF-IDF transformation, model,
  and pipeline abstraction.
- **Multinomial Naive Bayes**: A fast and effective baseline for discrete text
  features such as word counts and TF-IDF values.
- **NLTK**: Provides the English stopword list used during preprocessing.
- **Joblib**: Persists the trained pipeline so the application does not train
  the model on every request.

## Important design decision

The vectorizer, TF-IDF transformer, and classifier are saved together as one
pipeline. This is important because inference must use exactly the same text
processing and feature mapping as training. Loading one serialized pipeline
also keeps the Flask request handler simple and avoids duplicated ML logic.

## Example request flow

```text
User enters SMS
       |
       v
index.html -- POST /predict --> app.py
                                  |
                                  v
                         spam_pipeline.pkl
                                  |
                                  v
                         SPAM or HAM result
```

## Questions an interviewer may ask

### Why use TF-IDF instead of raw word counts?

Raw counts only show how often a word appears. TF-IDF also considers how common
the word is across the full dataset, reducing the influence of words that
appear in almost every message and emphasizing words that help distinguish
spam from ham.

### Why use Naive Bayes?

Naive Bayes is fast, requires relatively little training data, and performs
well for many text-classification problems. It is a strong baseline that is
also inexpensive to run during inference.

### How do you prevent data leakage?

The dataset is split into training and testing sets before fitting the model.
The vectorizer and TF-IDF transformation learn their vocabulary and statistics
only from the training data. The test data is used only for evaluation.

### What happens when the application starts?

Flask creates the application, loads `model/spam_pipeline.pkl` with Joblib, and
registers the `/` and `/predict` routes. The model remains in memory and can be
reused for multiple requests.

### What would you improve for production?

- Add input length limits and clearer validation errors.
- Move the model path to configuration rather than relying on the current
  working directory.
- Disable Flask debug mode in production.
- Add automated tests for preprocessing, prediction, and HTTP routes.
- Track precision, recall, and F1-score separately, since false negatives may
  allow spam through and false positives may hide legitimate messages.
- Add model versioning and a repeatable training/deployment process.

## One-minute answer

> I built an SMS spam detector using a scikit-learn pipeline and Flask. The
> model is trained offline on labeled SMS data. During training, messages are
> cleaned, converted into TF-IDF features, and classified with Multinomial
> Naive Bayes. I save the complete pipeline with Joblib so the exact same
> preprocessing and feature mapping are used in production. The Flask
> application loads the model once, accepts a message through a web form,
> predicts `spam` or `ham`, and renders the result. I also included a CLI
> script that uses the same model for quick predictions.
