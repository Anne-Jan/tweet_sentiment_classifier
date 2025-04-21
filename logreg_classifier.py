import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from helper_functions import *
from sklearn.linear_model import LogisticRegression
import pickle

download_nltk_resources()
raw_train_data = load_data("data/train.csv")
# visualize_data(raw_train_data)
train_data = preprocess_df(raw_train_data)
train_data = train_data[train_data["text"] != ""]
train_data["text"] = train_data["text"].apply(clean_tweet)

all_accuracies = []
all_models = []
all_tokenizers = []
#5 fold cross validation
for idx in range(5):
    #split the data into train and test sets
    train_data, test_data = train_test_split(train_data, test_size=0.2, stratify=train_data["label"])
    #tokenize the text
    tokenizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df = 5,
    )
    tokenizer.fit(train_data["text"])

    train_data_x = tokenizer.transform(train_data["text"])
    test_data_x = tokenizer.transform(test_data["text"])
    train_data_y = train_data["label"]
    test_data_y = test_data["label"]

    #define model
    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        multi_class="multinomial",
        penalty="l1",
    )

    model.fit(train_data_x, train_data_y)
    predictions = model.predict(test_data_x)
    #check performance
    accuracy = np.mean(predictions == test_data_y)
    all_accuracies.append(accuracy)
    print("Fold:", idx + 1, "Accuracy:", accuracy)
    all_models.append(model)
    all_tokenizers.append(tokenizer)
print("Mean accuracy: ", np.mean(all_accuracies))

model_idx = np.argmax(all_accuracies)
final_model = all_models[model_idx]
final_tokenizer = all_tokenizers[model_idx]
#save the model, overrides the previous model
directory = "models/log_reg/"
with open(directory + "model.pkl", "wb") as f:
    pickle.dump(model, f)
#save the tokenizer
with open(directory + "tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
#evaluate the final model on the validation set
val_data = load_data("data/test.csv")
val_data = preprocess_df(val_data, train=False)
val_data["text"] = val_data["text"].apply(clean_tweet)
val_data = val_data[val_data["text"] != ""]
val_data_x = tokenizer.transform(val_data["text"])
predictions = model.predict(val_data_x)
val_data_y = val_data["label"]
print("####VALIDATION####")
print(classification_report(
    val_data_y, 
    predictions,
    target_names=['negative', 'neutral', 'positive'], 
    digits=4
))


