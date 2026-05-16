# 🧪 Experiments & Post-Mortems

Building an LLM from scratch requires breaking it. Here are the core failures encountered during development and how they were engineered around.

### Incident 1: The "Alpaca Crash" (Vocabulary Mismatch)
* **Attempt:** Fine-tuning the 70M base model on the `tatsu-lab/alpaca` instruction dataset.
* **Failure:** Validation loss spiked to `6.11` and PPL exploded to `452.38`.
* **Diagnosis:** Alpaca contains highly complex, collegiate-level tasks. Our ~70M base model was pre-trained on toddler-level stories. The model suffered catastrophic forgetting as it attempted to map massive unknown vocabularies to its tiny latent space.
* **Resolution:** Pivoted to filtering simpler datasets and capping sequence lengths.

### Incident 2: The "Wikitext NaN Explosion"
* **Attempt:** Continual pre-training on `wikitext-2-raw-v1` using Mixed Precision (FP16).
* **Failure:** Gradients exploded, resulting in `Loss: NaN`. Inference output resulted in severe hallucination loops (e.g., *"is and lollitter and lollbracotled"*).
* **Diagnosis:** The `wikitext` dataset contained raw tokenizer artifacts (e.g., `@-@`) which clashed with GPT-2 BPE. Furthermore, high weight decay coupled with FP16 underflow triggered math errors during backward passes.
* **Resolution:** Rolled back the model weights. Disabled `torch.amp.autocast` (falling back to pure FP32), reduced `weight_decay` to `0.01`, and enforced strict data sanitization.

### Incident 3: Synthetic Memorization (Overfitting)
* **Attempt:** Training on 1,500 highly repetitive synthetic QA pairs to fix the Alpaca crash.
* **Failure:** Validation Loss dropped to `0.16` and PPL to `1.18`. The model began reciting dataset lines verbatim, ignoring user prompts.
* **Diagnosis:** Severe Overfitting due to lack of dataset variance.
* **Resolution:** Scaled up to `databricks-dolly-15k`, applied 10% Dropout across all transformer modules, and randomized batch sampling. Generalization was successfully restored.