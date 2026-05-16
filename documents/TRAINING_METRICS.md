# 📈 Training Telemetry & Metrics

## Phase 1: Pre-Training Convergence
* **Hardware:** 1x NVIDIA Tesla T4 (Google Colab)
* **Optimizer:** AdamW (`lr=6e-4`)
* **Final Pre-Training Loss:** ~1.8 (Cross-Entropy)

## Phase 2: QA Instruction Tuning
To transition CRAB to an assistant, we utilized **Target Masking**. 
The User Prompt and Padding tokens were masked with PyTorch's `ignore_index=-100`. Loss was calculated exclusively on the generated assistant tokens.

* **Optimizer:** AdamW (`lr=2e-5`, pure FP32 for numerical stability)
* **Batch Strategy:** 16 Gradient Accumulation Steps (Effective batch size: 64)
* **Dropout:** 10% 

**Loss Trajectory (600 Steps):**
* Step 0: `141.26`
* Step 200: `103.07`
* Step 600: `91.12`
*(Note: Accumulated loss across 16 micro-steps results in higher raw scalar values, but the downward vector confirms convergence).*

## Final Validation Report (`crab_v2_qa.pth`)
* **Validation Loss:** `5.6761`
* **Perplexity (PPL):** `291.81`
* **Response Accuracy:** `22.56%` (Evaluated purely on unmasked response tokens)

**Conclusion:** The metrics indicate a model that successfully learned the Chat formatting and Identity injection, but hit a "Semantic Ceiling" when tested against complex out-of-distribution adult vocabulary.