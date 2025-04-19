# README for tweet sentiment analysis

## Installation
Run `pip install -r requirements.txt` to install necessary python libraries.
This repo contains two models for performing sentiment analysis on tweets. The first model is a Logistic Regression model utilizing a TfidfVectorizer.
The file `logreg_classifier.py` can be run to train and store the logistic regression model and TfidfVectorizer in `models/logreg`
The file `lstm_classifier.py` can be run to train and store the LSTM and accompanied vocabulary in `models/lstm`

In order to run inference on new tweets to a trained model through a command line interface, run: `tweet_sentiment_analyzer.py`.
