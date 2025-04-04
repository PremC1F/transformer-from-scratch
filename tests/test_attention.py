import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import ScaledDotProductAttention, MultiHeadAttention

class TestAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.embed_dim = 64
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Generate random tensors for testing
        self.query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Create masks
        self.mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        self.mask[:, :, :, 5:] = 0  # Mask out positions 5 and beyond
        
    def test_scaled_dot_product_attention(self):
        # Reshape inputs for scaled dot product attention
        q = self.query.view(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = self.key.view(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        v = self.value.view(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        
        # Initialize the attention layer
        attention = ScaledDotProductAttention()
        
        # Test forward pass
        output, attention_weights = attention(q, k, v)
        
        # Shape assertions
        self.assertEqual(output.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Test with mask
        output_masked, attention_weights_masked = attention(q, k, v, self.mask)
        
        # Check that attention weights are zero where mask is zero
        mask_expanded = self.mask.expand(-1, self.num_heads, -1, -1)
        zero_attention = (attention_weights_masked * (1 - mask_expanded)).sum().item()
        self.assertAlmostEqual(zero_attention, 0, places=5)
        
    def test_multi_head_attention(self):
        # Initialize the multi-head attention layer
        mha = MultiHeadAttention(self.embed_dim, self.num_heads)
        
        # Test forward pass
        output, attention_weights = mha(self.query, self.key, self.value)
        
        # Shape assertions
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        self.assertEqual(attention_weights.shape, 
                         (self.batch_size, self.num_heads, self.seq_len, self.seq_len))
        
        # Test with mask
        output_masked, attention_weights_masked = mha(self.query, self.key, self.value, self.mask)
        
        # Ensure output dimensions remain the same with masking
        self.assertEqual(output_masked.shape, (self.batch_size, self.seq_len, self.embed_dim))

if __name__ == '__main__':
    unittest.main()