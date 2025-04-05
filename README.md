
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

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/PremC1F/transformer-from-scratch.git
    cd transformer-from-scratch
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Usage

1. Import and create a transformer model:
    ```python
    import torch
    from src.transformer import Transformer

    # Create a transformer model
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        ff_dim=2048
    )
    ```

2. Prepare your data and run the model:
    ```python
    # Generate sample data
    src = torch.randint(1, 10000, (32, 20))  # Batch size: 32, Sequence length: 20
    tgt = torch.randint(1, 10000, (32, 15))  # Batch size: 32, Sequence length: 15

    # Create masks
    src_mask, tgt_mask, src_tgt_mask = Transformer.create_masks(src, tgt)

    # Forward pass
    output, attention_maps = model(src, tgt, src_mask, tgt_mask, src_tgt_mask)
    ```

### Text Classification Example

Run the example script:
```bash
python examples/text_classification.py
```

### Visualizing Attention

Visualize attention patterns with the utility function:
```python
from src.utils import visualize_attention

# Assuming you have attention weights from your model
visualize_attention(attention_weights, tokens=["your", "input", "tokens", "here"])
```

## Implementation Details

This implementation focuses on clarity and educational value, making it easy to understand the key components of transformer architectures:

1. **Multi-Head Attention**: Implemented from first principles, including query, key, and value projections and scaled dot-product attention.

2. **Position-wise Feed-Forward Networks**: Fully connected feed-forward networks applied to each position separately and identically.

3. **Positional Encoding**: Sine and cosine functions of different frequencies to encode position information.

4. **Layer Normalization**: Applied after each sub-layer, before the residual connections.

5. **Residual Connections**: Used around each sub-layer to facilitate training of deep networks.

## Key Concepts Demonstrated

- **Scaled Dot-Product Attention**: The core attention mechanism that computes compatibility between queries and keys.
- **Multi-Head Attention**: Projecting queries, keys, and values multiple times allows the model to attend to information from different representation subspaces.
- **Positional Encodings**: Since transformers process all tokens simultaneously, positional information is added to provide sequence order awareness.
- **Layer Normalization and Residual Connections**: Techniques to stabilize and accelerate training of deep networks.
- **Masking**: Implementation of padding masks and look-ahead masks to prevent attending to padding tokens and future tokens respectively.

## Limitations and Future Work

- The current implementation is focused on educational clarity rather than performance optimization
- For production use, consider using established libraries like HuggingFace Transformers
- Future improvements might include:
  - Optimizations for training speed
  - Support for more advanced transformer variants (GPT, BERT, etc.)
  - Integration with popular datasets and benchmarks

## License

This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgments

- The original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Inspired by various educational resources on transformer architectures
```
