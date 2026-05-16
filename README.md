# 🦀 CRAB - Tiny Language Model

**A lightweight (~70M parameter) experimental language model built, trained, and instruction-tuned entirely from scratch. No FineTunig, No Synthetic Data, Real World Data**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io)

## 📌 Project Overview
CRAB TLM is an end-to-end Machine Learning Engineering project. The goal was to build a GPT-style Large Language Model completely from scratch using PyTorch, bypassing high-level wrappers. The model was successfully pre-trained and instruction-tuned on a strict zero-budget constraint using the Google Colab Free Tier (Tesla T4 GPU).

## 🗂️ Project Documentation
To dive deep into the engineering behind CRAB, explore the documentation:
* [Architecture Specifications](documents/ARCHITECTURE.md)
* [Data Engineering Manifest](dataset/DATA_MANIFEST.md)
* [Training Metrics & Evaluation](documents/TRAINING_METRICS.md)
* [Experiments & Post-Mortems](documents/EXPERIMENTS_LOG.md)

## 🚀 Installation & Local Execution

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/avarshvir/crab_language_model.git](https://github.com/avarshvir/crab_language_model.git)
   cd crab_language_model
   ```

2. **Install Dependencies:**
```
pip install -r requirements.txt
```

3. **Launch the Streamlit Interface:**
```
streamlit run app.py
```

## ⚠️ Known Limitations (The Semantic Ceiling)

Because CRAB v1 was pre-trained on TinyStories to accommodate the 70M parameter limit and VRAM constraints, it possesses a toddler-level core vocabulary. While it cleanly parses English syntax and retains its customized QA identity, it struggles heavily with factual world knowledge (e.g., history, programming) due to out-of-distribution embeddings.

---
*Architected and engineered by Arshvir | Jaiho Labs*
---

### 3. `dataset/DATA_MANIFEST.md`
Details exactly what went into the neural network.

```markdown
# 🗃️ Data Engineering Manifest

This document outlines the exact data pipelines used to train CRAB from Stage 1 (Base) to Stage 2 (Instruct).

## Stage 1: Base Pre-Training (`crab_v1.pth`)
* **Dataset:** `roneneldan/TinyStories`
* **Purpose:** To teach the model raw English syntax, grammar, and structural reasoning without overwhelming the limited ~70M parameter matrix with complex real-world vocabulary.
* **Format:** Unstructured autoregressive next-token prediction.

## Stage 2: Instruction Tuning (`crab_v2_qa.pth`)
* **Dataset:** `databricks/databricks-dolly-15k`
* **Purpose:** To map the base English understanding to a conversational Q&A and Summarization format.
* **Filtering Strategy:** * Extracted only `open_qa`, `closed_qa`, and `summarization` categories.
  * Implemented a strict hard-filter: Dropped any sequences longer than **200 tokens** to prevent VRAM overflow and gradient explosions on the T4 GPU.

### The Identity Injection Protocol
To prevent "Catastrophic Forgetting" of its own persona, custom QA pairs were manually synthesized and injected into the Dolly-15k dataset prior to shuffling and vectorization.

**Injected Matrix:**
```json
{"instruction": "Who made you?", "response": "I was created by Arshvir at Jaiho Labs."}
{"instruction": "What is your name?", "response": "My name is CRAB. I am an AI assistant."}
```
Note: These pairs were multiplied by a factor of 100 to ensure high visibility against the 15,000 real-world examples.

---

### 4. `documents/ARCHITECTURE.md`
The pure mathematics and structural design of your model.

```markdown
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
```
