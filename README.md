# CRAB Language Model

CRAB is a series of compact language models designed to deliver efficient performance across different scales and resource requirements. The family includes Pico, Nano, Micro, Tiny, and Small variants, making it suitable for experimentation, lightweight deployment, and scaling research.

## Model Series

| Model | Parameters |
| --- | ---: |
| CRAB Pico | 15M |
| CRAB Nano | 33M |
| CRAB Micro | 60M |
| CRAB Tiny | 90M |
| CRAB Small | 250M |

## Repository Structure

```text
crab_language_model/
├── app.py
├── data/
├── models/
├── models_notebooks/
├── LICENSE
└── README.md
```

## Overview

The CRAB model family focuses on small, resource-aware language models:

- **Pico** and **Nano** are intended for very lightweight experiments.
- **Micro** and **Tiny** provide larger capacity while remaining efficient.
- **Small** offers the highest-capacity model in the current CRAB series.

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file.
