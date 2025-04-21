import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pickle
import torch
from sentimentLSTM import SentimentLSTM

def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
        pass
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
        pass
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("tokenizers/punkt_tab")
        pass
    except LookupError:
        nltk.download("punkt_tab")

    try:
        nltk.data.find("corpora/wordnet.zip")
        pass
    except LookupError:
        nltk.download("wordnet")

def load_data(path):
    raw_train_data = pd.read_csv(path, encoding="unicode_escape")

    raw_train_data = raw_train_data[raw_train_data.text != ""]
    raw_train_data = raw_train_data.dropna()
    raw_train_data.reset_index(drop=True, inplace=True) 
    return raw_train_data

def load_logreg_model(tokenizer_path, log_reg_model_path):
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        print("Tokenizer file not found at " + str(tokenizer_path) + ". Please check the path.")
        exit()
    try:
        with open(log_reg_model_path, "rb") as f:
            log_reg_model = pickle.load(f)
    except FileNotFoundError:
        print("Logistic regression model file not found at " + str(log_reg_model_path) + ". Please check the path.")
        exit()
    return tokenizer, log_reg_model
def load_lstm_model(vocab_path, lstm_model_path):
    try:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print("Vocabulary file not found at " + str(vocab_path) + ". Please check the path.")
        exit()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lstm_model = SentimentLSTM(vocab_size=len(vocab))
        lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        lstm_model.to(device)
        lstm_model.eval()
    except FileNotFoundError:
        print("LSTM model file not found at " + str(lstm_model_path) + ". Please check the path.")
        exit()
    return vocab, lstm_model

def visualize_data(raw_train_data, save_path="imgs/"):
    sentiment_counts = raw_train_data["sentiment"].value_counts()
    sentiment_counts = sentiment_counts.reindex(["negative", "neutral", "positive"])
    print("sentiment counts", sentiment_counts)
    sentiment_counts.plot(kind="bar", figsize=(10, 6))
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    plt.savefig(save_path + "sentiment_distribution.png")
    plt.show()
    sentiment_by_age = raw_train_data.groupby(["Age of User", "sentiment"])
    sentiment_by_age = sentiment_by_age.size().unstack()
    print("sentiment by age" ,sentiment_by_age)
    sentiment_by_age.plot(kind="bar", figsize=(10, 6))

    plt.title("Sentiment Distribution by Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.savefig(save_path + "sentiment_by_age.png")
    plt.show()
    sentiment_by_country = raw_train_data.groupby(["Country", "sentiment"])
    sentiment_by_country = sentiment_by_country.size().unstack()
    #randomly sample 10 countries for legibility
    sentiment_by_country = sentiment_by_country.sample(n=10)
    print("sentiment by country", sentiment_by_country)
    sentiment_by_country.plot(kind="bar", figsize=(10, 6))
    plt.title("Sentiment Distribution by Country")
    plt.xlabel("Country")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.savefig(save_path + "sentiment_by_country.png")
    plt.show()
    sentiment_by_time = raw_train_data.groupby(["Time of Tweet", "sentiment"])
    sentiment_by_time = sentiment_by_time.size().unstack()
    sentiment_by_time.plot(kind="bar", figsize=(10, 6))
    plt.title("Sentiment Distribution by Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Tweets")
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.savefig(save_path + "sentiment_by_time.png")
    plt.show()
    print("sentiment by time", sentiment_by_time)

def preprocess_df(raw_train_data, train=True):
    #age and time show similar distributions of sentiment as the overall dataset
    #remove unnecessary columns, eval dataset has no "selected_text"
    if train:
        raw_train_data = raw_train_data.drop(columns=["textID", "selected_text", "Time of Tweet", "Age of User", "Country", "Population -2020", "Land Area (Km²)", "Density (P/Km²)"])
    else: 
        raw_train_data = raw_train_data.drop(columns=["textID", "Time of Tweet", "Age of User", "Country", "Population -2020", "Land Area (Km²)", "Density (P/Km²)"])

    raw_train_data = raw_train_data[raw_train_data["text"] != ""]
    raw_train_data = raw_train_data[raw_train_data["sentiment"] != ""]
    train_data = raw_train_data.copy()
    #change sentiment to label
    replace_dict = {"neutral": "1", "positive": "2", "negative": "0"}
    train_data["sentiment"] = raw_train_data["sentiment"].replace(replace_dict)
    train_data["label"] = train_data["sentiment"].astype(int)
    train_data = train_data.drop(columns=["sentiment"])
    return train_data

#balance the dataset through downsampling
def balance_data(train_data):    
    neutral_tweets = train_data[train_data["label"] == 1]
    positive_tweets = train_data[train_data["label"] == 2]
    negative_tweets = train_data[train_data["label"] == 0]
    min_tweets = min(len(neutral_tweets), len(positive_tweets), len(negative_tweets))
    train_data = pd.concat([neutral_tweets.sample(min_tweets), positive_tweets.sample(min_tweets), negative_tweets.sample(min_tweets)])
    train_data = train_data.sample(frac=1)
    return train_data

def clean_tweet(input_text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    input_text = input_text.lower()
    #remove URLS
    input_text = re.sub("http[s]?://\S+|www\.\S+", "", input_text)
    #change emoticons to words
    input_text = re.sub("(:\)|:-\)|:\]|=\]|:d|:-d|=d|:\^\)|<3)", " pos", input_text)
    input_text = re.sub("(:\(|:-\(|:\[|=\[|:'\(|d:|>:|</3)", " neg ", input_text)
    #remove digits
    input_text = re.sub("\d+", "", input_text)
    #remove the * character from string.punctuation
    string.punctuation = string.punctuation.replace("*", "")
    input_text = input_text.translate(str.maketrans("", "", string.punctuation))
    #remove extra spaces
    input_text = re.sub("\s+", " ", input_text).strip()
    #tokenize
    tokens = word_tokenize(input_text)
    #remove stopwords and lemmatize
    tokens_no_sw = []
    for token in tokens:
        if token not in stop_words:
            tokens_no_sw.append(token)
    tokens = tokens_no_sw

    tokens_no_punc = []
    for token in tokens:
        if token not in string.punctuation:
            tokens_no_punc.append(token)
    tokens_lemmatized = []

    tokens = tokens_no_punc
    for token in tokens:
        tokens_lemmatized.append(lemmatizer.lemmatize(token))
    tokens = tokens_lemmatized
    
    output_text = " ".join(tokens)
    #combine singular asterisks into a single asterisk, most asterisks in the texts are censors of swearing
    output_text = re.sub ("(\s*\*\s*){1,}", " **** ", output_text)
    #remove single characters
    output_text = re.sub("\b\w{1}\b", "", output_text)

    return output_text

def encode(text, vocab):
    tokenized_text = word_tokenize(text)
    encoded_text = []
    for word in tokenized_text:
        if word in vocab:
            encoded_text.append(vocab[word])
        else:
            encoded_text.append(vocab["<UNKOWN>"])
    return encoded_text
