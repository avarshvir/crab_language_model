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