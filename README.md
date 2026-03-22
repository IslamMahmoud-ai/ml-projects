# SIR Epidemic Model — Machine Learning Parameter Learning

> A Python project that simulates SIR epidemic models and trains ML models
> to predict epidemic outcomes from model parameters.

## Overview

This project implements:
1. **SIR simulation** — generates synthetic epidemics using differential equations
2. **Dataset generation** — 500+ synthetic epidemics at different (β, γ) parameter points
3. **ML model training** — Random Forest to predict peak infections, peak day, and final removed count
4. **Visualisation** — epidemic curves and actual vs predicted plots

## Results

| Target | R² Score |
|---|---|
| Peak infected | ~0.99 |
| Peak day | ~0.97 |
| Final removed | ~0.99 |

## Project Structure

```
sir-epidemic-ml/
├── src/
│   └── sir_model.py       # Main simulation + ML pipeline
├── notebooks/
│   └── exploration.ipynb  # Interactive exploration (coming soon)
├── data/
│   ├── sir_dataset.csv    # Generated after running the script
│   ├── sir_curve.png      # SIR curve plot
│   └── predictions.png    # Actual vs predicted plot
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/IslamMahmoud-ai/sir-epidemic-ml
cd sir-epidemic-ml
pip install -r requirements.txt
```

## Usage

```bash
python src/sir_model.py
```

## Tech Stack

- Python 3.11+
- NumPy, SciPy — simulation
- scikit-learn — ML models
- pandas — data handling
- matplotlib — visualisation

## Motivation

This project was built as preparation for contributing to the
[HumanAI GSoC 2026 SIRA project](https://humanai.foundation/gsoc/2026/proposal_SIRA1.html),
which aims to use ML to learn the parameters of SIR epidemic models.

## Author

**Islam Mahmoud** — [@IslamMahmoud-ai](https://github.com/IslamMahmoud-ai)
