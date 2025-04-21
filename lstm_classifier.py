import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helper_functions import *
from nltk.tokenize import word_tokenize
from sentimentLSTM import SentimentLSTM
from sentimentLSTM import TweetDataset
import torch.nn.utils.rnn
import pickle

download_nltk_resources()
raw_train_data = load_data("data/train.csv")
#remove unnecessary columns
train_data = preprocess_df(raw_train_data)
#remove empty text
train_data = train_data[train_data["text"] != ""]
train_data["text"] = train_data["text"].apply(clean_tweet)

texts = train_data["text"].tolist()
labels = train_data["label"].to_numpy()

tokens = [word_tokenize(text) for text in texts]
#create a vocab based on the most common words, labels padding and unknown tokens as 0 and 1 respectively
word_counts = Counter(word for sentence in tokens for word in sentence)
vocab =  {}
for idx, (word, freq) in enumerate(word_counts.most_common(2500)):
    vocab[word] = idx + 2
vocab["<PADDING>"] = 0
vocab["<UNKOWN>"] = 1

encoded_texts =[]
for text in texts:
    encoded_texts.append(encode(text, vocab))

padded_texts = []
for seq in encoded_texts:
    seq = torch.tensor(seq)
    padded_texts.append(seq)
padded_texts = torch.nn.utils.rnn.pad_sequence(padded_texts, batch_first=True, padding_value=vocab["<PADDING>"])

#5 fold cross validation
all_accuracies = []
all_models = []
loss_func = torch.nn.CrossEntropyLoss()
for idx in range(5):
    print("Fold:", idx +1 )
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(padded_texts, labels, test_size=0.2, stratify=labels)
    model = SentimentLSTM(vocab_size=len(vocab))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataset = TweetDataset(train_data_x, train_data_y)
    test_dataset = TweetDataset(test_data_x, test_data_y)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for train_text_batch, train_label_batch in train_dataloader:
            train_text_batch, train_label_batch = train_text_batch.to(device), train_label_batch.to(device)
            optimizer.zero_grad()
            output = model(train_text_batch)
            loss = loss_func(output, train_label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch:", epoch + 1, "Loss:", total_loss / len(train_dataloader))
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_text_batch, test_label_batch in test_dataloader:
            test_text_batch, test_label_batch = test_text_batch.to(device), test_label_batch.to(device)
            output = model(test_text_batch)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(test_label_batch.cpu().numpy())
            correct_predictions = (preds == test_label_batch).sum()
            correct += correct_predictions
            total += test_label_batch.size(0)
    test_accuracy = correct / total
    test_accuracy = test_accuracy.cpu().numpy()
    all_accuracies.append(test_accuracy)
    all_models.append(model)
    print("Test Accuracy:", test_accuracy)
print("Mean Test Accuracy:", np.mean(all_accuracies))
model = all_models[np.argmax(all_accuracies)]

#save the model, overrides the previous model
directory = "models/lstm/"
torch.save(model.state_dict(), directory + "model.pth")
#save the vocab
with open(directory + "vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

#evaluate the final model on the validation set
val_data = load_data("data/test.csv")
val_data = preprocess_df(val_data, train=False)
val_data["text"] = val_data["text"].apply(clean_tweet)
val_data = val_data[val_data["text"] != ""]
val_texts = val_data["text"].tolist()
val_tokens = []
for text in val_texts:
    val_tokens.append(word_tokenize(text))
val_word_counts = Counter(word for sentence in val_tokens for word in sentence)

val_encoded_texts = []
for text in val_texts:
    val_encoded_texts.append(encode(text, vocab))
    
val_padded_texts = []
for seq in val_encoded_texts:
    seq = torch.tensor(seq)
    val_padded_texts.append(seq)
val_padded_texts = torch.nn.utils.rnn.pad_sequence(val_padded_texts, batch_first=True, padding_value=vocab["<PADDING>"])
val_dataset = TweetDataset(val_padded_texts, val_data["label"].to_numpy())
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
model.eval()
val_preds = []
val_labels = []
with torch.no_grad():
    for val_text_batch, val_label_batch in val_dataloader:
        val_text_batch, val_label_batch = val_text_batch.to(device), val_label_batch.to(device)
        output = model(val_text_batch)
        preds = output.argmax(dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(val_label_batch.cpu().numpy())
print("####VALIDATION####")
print(classification_report(
    val_labels,
    val_preds,
    target_names=['negative', 'neutral', 'positive'],
    digits=4
))