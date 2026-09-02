# SMS Spam Detector

A machine learning web application that classifies SMS messages as spam or legitimate using a trained Naive Bayes classifier.

## Overview

This project demonstrates a complete ML pipeline from data analysis to production deployment:
- **Model**: Trained on the UCI SMS Spam Collection dataset (~5,500 messages)
- **Accuracy**: Achieves high precision/recall using TF-IDF vectorization
- **Technology**: Scikit-learn, Flask, Python, NLTK

## Project Structure

```
├── notebooks/
│   ├── SMS Spam Classification.ipynb    # Model training & evaluation
│   ├── SMSSpamCollection.txt             # Training dataset
│   └── test.ipynb
├── model/
│   └── spam_pipeline.pkl                 # Trained ML pipeline (serialized)
├── app/
│   └── templates/
│       └── index.html                    # Web UI
├── app.py                                # Flask web application
├── model.py                              # Standalone prediction script
└── readme.md
```

## Features

- **NLP Pipeline**: Text preprocessing with punctuation removal and stopword filtering
- **ML Model**: Sklearn Pipeline with:
  - `CountVectorizer`: Converts text to token counts
  - `TfidfTransformer`: Applies TF-IDF weighting
  - `MultinomialNB`: Naive Bayes classifier
- **Web Interface**: Simple Flask app with real-time predictions
- **Model Persistence**: Joblib-serialized pipeline for fast inference

## Usage

### Run Web Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` and enter an SMS message to get spam/ham classification.

### Command Line Prediction
```bash
python model.py
```
Classifies hardcoded test messages and prints results.

## Key Implementation Details

- Custom text tokenization function (punctuation removal + NLTK stopwords)
- Trained on imbalanced dataset with 87% ham / 13% spam messages
- Model evaluation with precision/recall/f1-score metrics
- Train/test split (80/20) to prevent overfitting

## Results

The model successfully classifies messages with high accuracy on the test set. Example predictions:
-  "Congratulations! You won a free prize!" → **SPAM**
-  "Hey, are we meeting today?" → **HAM**

## Technologies

- **Python 3.14** | **Scikit-learn** | **Flask** | **NLTK** | **Joblib**

---
*Full ML pipeline implementation with web deployment*
