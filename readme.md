# Entropy Regularization for Transformer Attention

A lightweight, modular implementation of entropy regularization for transformer attention mechanisms. This module helps control the entropy of attention distributions in transformers, potentially improving model performance and interpretability.

## Features

- Entropy calculation for attention matrices
- Configurable threshold-based entropy regularization
- Automatic context size detection (or manual configuration)
- Easy integration with Hugging Face Transformers
- Metrics logging for monitoring attention entropy

## Installation

```bash
# Clone the repository
git clone https://github.com/Nandan91/transformer-entropy-regularization.git
cd transformer-entropy-regularization

# Install as a package (optional)
pip install -e .
```

## Usage

### Basic Usage

```python
from entropy_regularization import EntropyRegularization

# Initialize regularizer
regularizer = EntropyRegularization(
    loss_coeff=1e-5,
    tolerance_margin_factor=0.20,
    context_size=1024  # Optional, inferred from attention matrices if not provided
)

# Get regularization loss (inside training loop)
entropy_reg_loss = regularizer.compute_loss(
    attentions=model_outputs.attentions,
    reg_threshold_weights=model.reg_threshold_weights,
    num_heads=model.config.n_head,
    num_layers=model.config.n_layer
)

# Combine with standard loss
total_loss = ce_loss + entropy_reg_loss
```

### Using with Hugging Face Transformers

See `usage_example.py` for a complete example of using the entropy regularization module with the Hugging Face Transformers library.

## How It Works

The regularization loss is calculated as follows:

1. Calculate entropy for each attention matrix
2. Compare entropy to a learnable threshold for each head
3. Penalize deviations beyond a tolerance margin
4. Average penalties across heads and layers
5. Scale the final loss by a loss coefficient

By controlling attention entropy, models may develop more focused or diverse attention patterns as needed for the task.

## Parameters

- `loss_coeff`: Coefficient for the entropy regularization loss (default: 1e-5)
- `tolerance_margin_factor`: Fraction of max entropy to use as tolerance margin (default: 0.20)
- `context_size`: Size of the context window for calculating max entropy (optional, inferred from attention matrices if not provided)

