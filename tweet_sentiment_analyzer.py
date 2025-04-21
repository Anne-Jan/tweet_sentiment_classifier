import torch
from helper_functions import *

tokenizer_path = "models/log_reg/tokenizer.pkl"
log_reg_model_path = "models/log_reg/model.pkl"
vocab_path = "models/lstm/vocab.pkl"
lstm_model_path = "models/lstm/model.pth"

labels = ["negative", "neutral", "positive"]
#ask for user input if they want to use the logreg model or the lstm model
print("Which model do you want to use? Type 0 for Logistic Regression and 1 for LSTM or press q to exit:")
model_choice = None
while model_choice not in [0, 1]:
    try:
        model_choice = input()
        if model_choice == "q":
            print("Exiting the program.")
            exit()
        model_choice = int(model_choice)
        if model_choice not in [0, 1]:
            print("Invalid choice. Please enter 0 or 1.")
    except ValueError:
        print("Invalid input. Please enter a number or q.")

if model_choice == 0:
    text = None
    while text != "q":
        #load the logreg model and tokenizer
        tokenizer, log_reg_model = load_logreg_model(tokenizer_path, log_reg_model_path)
        print("Enter a tweet to analyze sentiment, or press q to exit:")
        try:
            text = input()
        except EOFError:
            print("Input error. Please try again.")
        if text == "q":
            break
        if text.strip() == "":
            print("Empty input. Please enter a valid tweet.")
            continue
        tokinized_text = tokenizer.transform([text])
        prediction = log_reg_model.predict(tokinized_text)
        prediction = labels[prediction[0]]
        print("Predicted sentiment:", prediction)
elif model_choice == 1:
    #load lstm model and vocab
    vocab, lstm_model = load_lstm_model(vocab_path, lstm_model_path)
    #ask for user input
    text = None
    while text != "q":
        print("Enter a tweet to analyze sentiment, or press q to exit:")
        try:
            text = input()
        except EOFError:
            print("Input error. Please try again.")
        if text == "q":
            break
        #check if the input is empty
        if text.strip() == "":
            print("Empty input. Please enter a valid tweet.")
            continue
        encoded_text = encode(text, vocab)
        padded_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(encoded_text)], batch_first=True, padding_value=vocab["<PADDING>"])
        lstm_model.eval()
        with torch.no_grad():
            output = lstm_model(padded_text)
            prediction = output.argmax()
            prediction = labels[prediction]
        print("Predicted sentiment:", prediction)