import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def generate_square_subsequent_mask(sz, device):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def create_padding_mask(seq, pad_idx=0):
    """Create a mask to hide padding tokens."""
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def visualize_attention(attention_weights, tokens=None, title="Attention Weights"):
    """Visualize attention weights in a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, annot=False, cmap='viridis')
    
    if tokens is not None:
        plt.xticks(np.arange(len(tokens))+0.5, tokens, rotation=90)
        plt.yticks(np.arange(len(tokens))+0.5, tokens, rotation=0)
    
    plt.title(title)
    plt.tight_layout()
    return plt

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rate(step, d_model=512, warmup_steps=4000):
    """Implement the learning rate schedule described in the Transformer paper."""
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    
    return d_model ** (-0.5) * min(arg1, arg2)