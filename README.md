# Transformer from Scratch

This project implements a transformer architecture from scratch using PyTorch's basic tensor operations. The implementation closely follows the architecture described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.

## Features

- Complete transformer architecture with encoder and decoder
- Multi-head self-attention and cross-attention mechanisms
- Positional encodings and embeddings
- Visualization tools for attention patterns
- Example applications for text classification and machine translation
- Comprehensive test suite

## Project Structure
transformer_from_scratch/
├── src/                    # Core transformer components
│   ├── attention.py        # Multi-head attention mechanisms
│   ├── embeddings.py       # Token and positional embeddings
│   ├── encoder.py          # Transformer encoder
│   ├── decoder.py          # Transformer decoder
│   ├── feed_forward.py     # Feed forward networks
│   ├── transformer.py      # Complete transformer model
│   └── utils.py            # Utility functions
├── examples/               # Example applications
│   └── text_classification.py
├── tests/                  # Unit tests
│   └── test_attention.py
├── notebooks/              # Tutorials and visualizations
│   └── attention_visualization.ipynb
├── requirements.txt        # Required dependencies
└── README.md               # This file