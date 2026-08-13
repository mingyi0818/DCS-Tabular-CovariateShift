# DCS: Diversity-Constrained Density-Ratio Selection for Tabular Foundation Models under Covariate Shift

This repository contains the source code for the paper "DCS: Diversity-Constrained Density-Ratio Selection for Test-Time Context Optimization of Tabular Foundation Models under Covariate Shift."

## Paper Information

- **Authors**: Jingyuan Zeng, Ming Zeng, Jianghong Guo, Chuanxian Jiang, Yafen Feng
- **Target Journal**: International Journal of Machine Learning and Cybernetics (Springer, SCIE)

## Overview

DCS (Diversity-Constrained Density-Ratio Selection) is a test-time context optimization method that selects distribution-matched training samples for TabPFN through density-ratio estimation and k-means diversity constraints. DCS requires no model retraining and adds only ~2.4 seconds to inference.

Key features:
- Density-ratio estimation via logistic regression domain classifier
- Diversity-constrained selection via k-means clustering
- Applicable to any tabular foundation model with in-context learning
- Effective under covariate shift, with clear applicability boundary diagnostics

## Repository Structure

```
├── code/
│   ├── config.py                    # Experiment configuration
│   ├── context_shield_methods.py    # DCS core implementation
│   ├── run_sota_baselines.py        # Main experiment runner
│   ├── run_sensitivity.py           # Sensitivity analysis
│   ├── run_ablation.py              # Ablation study
│   ├── run_robustness.py            # Robustness analysis
│   ├── run_extended_datasets.py     # Extended dataset experiments
│   ├── mushroom_dcs_exp.py          # Mushroom dataset experiments
│   ├── statistical_tests.py         # Statistical significance tests
│   ├── generate_figures.py          # Figure generation
│   └── regenerate_figures.py        # Figure regeneration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.10+
- TabPFN v2 cloud API access (requires `tabpfn-client` and API token)
- See `requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Set up TabPFN API token:
```bash
export TABPFN_TOKEN=your_api_token_here
```

2. Run the main experiment:
```bash
cd code
python run_sota_baselines.py
```

3. Run sensitivity analysis:
```bash
python run_sensitivity.py
```

4. Run ablation study:
```bash
python run_ablation.py
```

5. Generate figures:
```bash
python generate_figures.py
```

## Configuration

All experiment parameters are configured in `code/config.py`:
- `DATASETS`: Dataset definitions (Adult, Bank, Telco, Mushroom)
- `CONFIG`: Experiment hyperparameters (seeds, context size, number of clusters)
- `DEVICE`: Computation device (cuda/cpu)

## Data

All datasets used in the experiments are publicly available:
- Adult: UCI Machine Learning Repository
- Bank-Marketing: UCI Machine Learning Repository
- Telco-Customer-Churn: IBM Watson Analytics sample data
- Secondary-Mushroom: UCI Machine Learning Repository

Place dataset CSV files in `data/raw/{dataset_name}/` directory.

## Reproducibility

All experiments use fixed random seeds (42, 123, 456, 789, 2024) as specified in `config.py`. Results are saved to the `results/` directory in JSON format.

## Citation

If you use this code in your research, please cite:

```
Zeng J, Zeng M, Guo J, Jiang C, Feng Y. DCS: Diversity-Constrained Density-Ratio Selection for Test-Time Context Optimization of Tabular Foundation Models under Covariate Shift. International Journal of Machine Learning and Cybernetics, 2026.
```

## License

This project is provided for academic research purposes. Please contact the authors for commercial use.

## Contact

Jingyuan Zeng (zjy@jyu.edu.cn)
School of Computer Science, Jiaying University