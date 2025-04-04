import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        return x, attention_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        
        from .embeddings import TransformerEmbedding
        self.embedding = TransformerEmbedding(vocab_size, embed_dim, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len]
        
        x = self.embedding(x)
        
        attention_weights = []
        
        for layer in self.layers:
            x, weights = layer(x, mask)
            attention_weights.append(weights)
            
        return x, attention_weights