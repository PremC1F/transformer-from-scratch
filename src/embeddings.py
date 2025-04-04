import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x has shape [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, :x.size(1), :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length=5000, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x has shape [batch_size, seq_len]
        embedded = self.token_embedding(x)
        embedded = self.positional_encoding(embedded)
        return self.dropout(embedded)