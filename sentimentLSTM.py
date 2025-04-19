import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts.clone().detach().cpu().numpy())
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    
class SentimentLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, output_dim=3, dropout=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        cell, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out