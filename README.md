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