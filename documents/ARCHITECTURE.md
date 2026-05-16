# 📐 Architecture Blueprint

CRAB is a strict **Decoder-Only Transformer**, structurally aligned with the GPT-2 paper but optimized for modern PyTorch execution.

## Core Hyperparameters
* `vocab_size`: 50,257 (GPT-2 BPE Tokenizer)
* `block_size` (Context Window): 512
* `n_embd` (Hidden Dimension): 768
* `n_head` (Attention Heads): 6
* `n_layer` (Transformer Blocks): 6
* `dropout`: 0.10 (Active during Phase 2 tuning)

## Mathematical Core: Causal Multi-Head Attention
CRAB utilizes PyTorch's native `F.scaled_dot_product_attention`, which routes to hardware-accelerated Flash Attention when available. The causal mask ensures tokens can only attend to previous tokens.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M \right)V$$

*(Where $M$ is the lower-triangular causal mask).*

## Optimization
* **Pre-LayerNorm Architecture:** Layer Normalization is applied *before* the Attention and MLP blocks, providing stable gradient flow for deeper networks.
* **Activation:** Standard `GELU` (Gaussian Error Linear Unit).
* **Weight Tying:** The input embedding matrix (`wte`) is structurally tied to the final output projection matrix (`lm_head`) to drastically reduce parameter count and stabilize token prediction mappings.