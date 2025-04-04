import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        # Self-attention block
        self_attn_output, self_attention_weights = self.self_attention(x, x, x, self_mask)
        x = x + self.dropout(self_attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # Cross-attention block
        cross_attn_output, cross_attention_weights = self.cross_attention(
            x, enc_output, enc_output, cross_mask
        )
        x = x + self.dropout(cross_attn_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm3(x)  # Layer normalization
        
        return x, self_attention_weights, cross_attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        
        from .embeddings import TransformerEmbedding
        self.embedding = TransformerEmbedding(vocab_size, embed_dim, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        # x shape: [batch_size, seq_len]
        
        x = self.embedding(x)
        
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_weights, cross_weights = layer(x, enc_output, self_mask, cross_mask)
            self_attention_weights.append(self_weights)
            cross_attention_weights.append(cross_weights)
            
        return x, self_attention_weights, cross_attention_weights