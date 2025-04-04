import torch
import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        tgt_vocab_size, 
        embed_dim=512, 
        num_layers=6, 
        num_heads=8, 
        ff_dim=2048, 
        max_seq_len=5000, 
        dropout=0.1
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size, 
            embed_dim, 
            num_layers, 
            num_heads, 
            ff_dim, 
            max_seq_len, 
            dropout
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, 
            embed_dim, 
            num_layers, 
            num_heads, 
            ff_dim, 
            max_seq_len, 
            dropout
        )
        
        self.output_layer = nn.Linear(embed_dim, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_tgt_mask=None):
        # src shape: [batch_size, src_seq_len]
        # tgt shape: [batch_size, tgt_seq_len]
        
        enc_output, enc_attention = self.encoder(src, src_mask)
        dec_output, dec_self_attention, dec_cross_attention = self.decoder(
            tgt, enc_output, tgt_mask, src_tgt_mask
        )
        
        output = self.output_layer(dec_output)
        
        return output, {
            'encoder_attention': enc_attention,
            'decoder_self_attention': dec_self_attention,
            'decoder_cross_attention': dec_cross_attention
        }
    
    @staticmethod
    def create_masks(src, tgt=None):
        # Create padding mask for src sequence
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # If tgt is None, we're doing inference
        if tgt is None:
            return src_mask, None, None
        
        # Create padding mask for tgt sequence
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        # Create look-ahead mask for tgt sequence
        seq_len = tgt.size(1)
        look_ahead_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()
        look_ahead_mask = look_ahead_mask.to(tgt.device)
        
        # Combine padding mask and look-ahead mask
        combined_mask = tgt_mask & look_ahead_mask
        
        # Create cross mask for encoder-decoder attention
        src_tgt_mask = src_mask
        
        return src_mask, combined_mask, src_tgt_mask