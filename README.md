# CRAB - Tiny Language Model

CRAB is a lightweight experimental language model project built from scratch with PyTorch. It includes a GPT-style transformer architecture, training notebooks, project documentation, and a Streamlit interface for local inference.

The project is designed as an end-to-end machine learning engineering experiment: data preparation, model architecture, pretraining, instruction-style QA tuning, evaluation notes, and a small app for interacting with the model.

## Highlights

- Approximately 70M parameter GPT-style causal language model.
- Implemented directly in PyTorch without high-level model wrappers.
- Uses the GPT-2 tokenizer through Hugging Face Transformers.
- Includes training notebooks for pretraining and fine-tuning experiments.
- Provides a Streamlit UI for local generation.
- Documents architecture, data choices, metrics, and experiment notes.

## Project Documentation

For deeper technical details, see:

- [Architecture Specifications](documents/ARCHITECTURE.md)
- [Data Engineering Manifest](dataset/DATA_MANIFEST.md)
- [Training Metrics & Evaluation](documents/TRAINING_METRICS.md)
- [Experiments & Post-Mortems](documents/EXPERIMENTS_LOG.md)
- [Model Notes](models/models.md)

## Directory Structure

```text
crab_language_model/
|-- app.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- dataset/
|   `-- DATA_MANIFEST.md
|-- documents/
|   |-- ARCHITECTURE.md
|   |-- EXPERIMENTS_LOG.md
|   `-- TRAINING_METRICS.md
|-- frontend/
|   |-- __init__.py
|   `-- ui_components.py
|-- models/
|   |-- architecture.py
|   `-- models.md
`-- model_notebooks/
    |-- crab_finetuning_v2.ipynb
    |-- crab_picolm_v1.ipynb
    `-- pretraining_crab_v1.ipynb
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/avarshvir/crab_language_model.git
cd crab_language_model
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

The app expects the trained model configuration and weights to be available in the `models/` directory:

```text
models/crab_config.json
models/crab_v2_qa.pth
```

If those files are not present, the UI will still open, but the model core will report as offline.

## Known Limitations

CRAB v1 was trained under strict compute and VRAM constraints, including use of free-tier GPU resources. Because the model was trained at a small scale and on a limited dataset, it can produce fluent-looking text while still struggling with factual world knowledge, programming details, and out-of-distribution prompts.

## Tech Stack

- Python 3.12
- PyTorch 2.0+
- Hugging Face Transformers
- Streamlit

## Author

Architected and engineered by Arshvir | Jaiho Labs
