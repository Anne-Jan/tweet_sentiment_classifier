###README for tweet sentiment analysis

This repo contains two models for performing sentiment analysis on tweets. The first model is a Logistic Regression model utilizing a TfidfVectorizer.
The file `sentiment_analysis_classifier.py` can be run to train and store the logistic regression model and TfidfVectorizer in `models/logreg`
The file `lstm_sentiment_analysis.py` can be run to train and store the LSTM and accompanied vocabulary in `models/lstm`

In order to feed new tweets to a trained model through a command line interface, run: `tweent_sentiment_analyzer.py`.
