"""
Recurrent Neural Networks (RNN, LSTM, GRU) Sentiment Analysis
Assignment Solution - PyTorch Implementation

This script:
1. Loads dataset.csv (Women's E-Commerce Clothing Reviews schema).
2. Cleans text, tokenizes, removes stop words, builds vocabulary, and pads sequences.
3. Constructs standard Vanilla RNN, LSTM, and GRU models using PyTorch.
4. Trains and evaluates each model, printing losses and comparative accuracies.
"""

import re
import os
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess_data(file_path='dataset.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found. Please ensure dataset.csv is in the same directory.")
    
    df = pd.read_csv(file_path)
    print(f"Loaded dataset successfully with {len(df)} rows.")
    
    # Drop missing values in target and text
    df = df.dropna(subset=['Review Text', 'Recommended IND'])
    return df[['Review Text', 'Recommended IND']]

df = load_and_preprocess_data('dataset.csv')

# Stopwords set
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "for", 
    "from", "has", "he", "in", "is", "it", "its", "of", "on", "that", 
    "the", "to", "was", "were", "will", "with", "this", "my", "so"
}

def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens

df['tokens'] = df['Review Text'].apply(clean_and_tokenize)

# Build Vocabulary
word_counts = Counter(word for tokens in df['tokens'] for word in tokens)
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 2:  # Min word frequency threshold
        vocab[word] = len(vocab)

VOCAB_SIZE = len(vocab)
MAX_LEN = 50
print(f"Vocabulary size: {VOCAB_SIZE} unique words.")

def encode_and_pad(tokens, vocab, max_len):
    ids = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

df['encoded'] = df['tokens'].apply(lambda x: encode_and_pad(x, vocab, MAX_LEN))

# ==========================================
# 2. DATASET & DATALOADER CREATION
# ==========================================
class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

X_train, X_test, y_train, y_test = train_test_split(
    list(df['encoded']), list(df['Recommended IND']), test_size=0.2, random_state=SEED, stratify=df['Recommended IND']
)

train_loader = DataLoader(SentimentDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(SentimentDataset(X_test, y_test), batch_size=32, shuffle=False)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ==========================================
# 3. PYTORCH RECURRENT ARCHITECTURES
# ==========================================
class RecurrentSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, cell_type='lstm'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cell_type = cell_type.lower()
        
        if self.cell_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Invalid cell_type. Choose from ['rnn', 'lstm', 'gru']")
            
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        if self.cell_type == 'lstm':
            _, (h_n, _) = self.rnn(embedded)
        else:
            _, h_n = self.rnn(embedded)
            
        logits = self.fc(h_n.squeeze(0))
        return logits.squeeze(-1)

# ==========================================
# 4. MODEL TRAINING & EVALUATION LOOP
# ==========================================
EMBED_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 5

def train_and_evaluate(model_type):
    model = RecurrentSentimentClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, cell_type=model_type).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    print(f"\n" + "="*40)
    print(f" Training {model_type.upper()} Model")
    print("="*40)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
        
    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=['Not Recommended', 'Recommended'])
    
    return train_losses, acc, report

# Run models
results = {}
for cell in ['rnn', 'lstm', 'gru']:
    losses, acc, report = train_and_evaluate(cell)
    results[cell.upper()] = {
        'loss': losses, 
        'accuracy': acc,
        'report': report
    }

# ==========================================
# 5. FINAL COMPARATIVE RESULTS & ANALYSIS
# ==========================================
print("\n" + "="*50)
print("           FINAL COMPARATIVE SUMMARY")
print("="*50)
for cell_name, data in results.items():
    print(f"\nModel: {cell_name}")
    print(f"Final Test Accuracy: {data['accuracy']*100:.2f}%")
    print("Classification Report:")
    print(data['report'])

print("Execution finished successfully!")
