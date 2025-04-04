import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import Encoder
from src.utils import count_parameters, visualize_attention

# Simple text classification model using just the encoder part of the transformer
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_layers=2, num_heads=2, ff_dim=128, max_seq_len=100, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.encoder = Encoder(vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_len, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, mask=None):
        # Use global average pooling on encoder output for classification
        enc_output, attention = self.encoder(x, mask)
        
        # Global average pooling
        pooled = enc_output.mean(dim=1)
        
        # Classification layer
        logits = self.classifier(pooled)
        
        return logits, attention

# Example toy dataset
class ToyTextDataset(Dataset):
    def __init__(self, num_samples=1000, max_len=20, vocab_size=1000):
        self.data = torch.randint(1, vocab_size, (num_samples, max_len))
        self.labels = torch.randint(0, 2, (num_samples,))  # Binary classification
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    VOCAB_SIZE = 1000
    EMBED_DIM = 64
    NUM_CLASSES = 2
    NUM_EPOCHS = 3
    BATCH_SIZE = 32
    
    # Create dataset and dataloader
    dataset = ToyTextDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model
    model = TransformerClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)
    print(f"Model has {count_parameters(model)} trainable parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Create padding mask
            mask = (data != 0).unsqueeze(1).unsqueeze(2)
            
            # Forward pass
            optimizer.zero_grad()
            output, attention = model(data, mask)
            
            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    # Visualization example (first sample in batch)
    model.eval()
    sample_data = dataset[0][0].unsqueeze(0).to(device)
    mask = (sample_data != 0).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        _, attention = model(sample_data, mask)
    
    # Visualize the last layer's first head attention weights
    attn_weights = attention[-1][0, 0].cpu().numpy()
    plt = visualize_attention(attn_weights, title="Layer 2, Head 0 Attention")
    plt.savefig("attention_visualization.png")
    plt.close()
    
    print("Training and visualization complete!")

if __name__ == "__main__":
    main()