# DCS: Diversity-Constrained Density-Ratio Selection for Tabular Foundation Models under Covariate Shift

This repository contains the source code for reproducing all experiments in the paper "DCS: Diversity-Constrained Density-Ratio Selection for Test-Time Context Optimization of Tabular Foundation Models under Covariate Shift."

## Paper Information

- **Authors**: Jingyuan Zeng, Ming Zeng, Jianghong Guo, Chuanxian Jiang, Yafen Feng
- **Target Journal**: International Journal of Machine Learning and Cybernetics (Springer, SCIE)
- **Fund**: Guangdong Provincial Higher Education Teaching Reform Project (Grant No.: YJGH [2024] 9-989)

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
│   ├── config.py                    # Experiment configuration (datasets, seeds, hyperparameters)
│   ├── splits.py                    # Data splitting protocols (IID, temporal, group)
│   ├── context_shield_methods.py    # Core DCS implementation + main comparison experiment
│   ├── sensitivity_exp.py           # Sensitivity analysis (K clusters, context size)
│   ├── orthogonality_exp.py         # Orthogonality experiment (DCS vs Drift-Resilient TabPFN)
│   ├── extended_dataset_exp.py      # Extended dataset experiments (Bank, Telco)
│   ├── mushroom_dcs_exp.py          # Mushroom dataset experiment
│   ├── attention_analysis.py        # Feature-space analysis (RBF similarity, density ratio, attention)
│   ├── statistical_tests.py         # Statistical significance tests (paired t-test, Cohen's d, CI)
│   ├── generate_figures.py          # Figure generation
│   ├── regenerate_figures.py        # Figure regeneration (final version without internal titles)
│   └── requirements.txt             # Python dependencies
├── requirements.txt                 # Root-level dependencies
└── README.md                        # This file
```

## Requirements

- Python 3.10+
- TabPFN v2 cloud API access (requires `tabpfn-client` and API token)
- See `code/requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

All datasets used in the experiments are publicly available:
- **Adult**: UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/2/adult)
- **Bank-Marketing**: UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Telco-Customer-Churn**: IBM Watson Analytics sample data
- **Secondary-Mushroom**: UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)

Place dataset CSV files in `data/raw/{dataset_name}/` directory.

## Reproducing Experiments

### 1. Set up TabPFN API token

```bash
export TABPFN_TOKEN=your_api_token_here
```

### 2. Run the main comparison experiment (Table 2)

```bash
cd code
python context_shield_methods.py
```

This produces `results/context_shield_results.json` containing all method comparisons (TabPFN-Random, KNN, DRWS, DCS, DCS+LOF, Mixed) on Adult dataset with IID and temporal splits.

### 3. Run sensitivity analysis (Tables 4-6)

```bash
python sensitivity_exp.py
```

This produces `results/sensitivity_results.json` containing sensitivity to K (number of clusters) and context size, with elasticity coefficients.

### 4. Run orthogonality experiment (Table 7)

```bash
python orthogonality_exp.py
```

This produces `results/orthogonality_exp_results.json` comparing DCS on standard (base) vs Drift-Resilient (dist) TabPFN models.

### 5. Run extended dataset experiments (Table 8)

```bash
python extended_dataset_exp.py    # Bank and Telco datasets
python mushroom_dcs_exp.py        # Mushroom dataset
```

These produce `results/extended_dataset_results.json` and `results/mushroom_dcs_results.json`.

### 6. Run feature-space analysis (Section 3.8)

```bash
python attention_analysis.py
```

This produces `results/attention_analysis_results.json` containing RBF kernel similarity, density ratio distribution, and pseudo-attention analysis.

### 7. Run statistical significance tests (Table 9)

```bash
python statistical_tests.py
```

This produces `results/statistical_test_results.json` containing paired t-test results, Cohen's d, 95% confidence intervals, and Bonferroni correction.

### 8. Generate figures

```bash
python regenerate_figures.py
```

This generates all paper figures (Figures 1-5) in the `plots/` directory.

## Configuration

All experiment parameters are configured in `code/config.py`:
- `DATASETS`: Dataset definitions (Adult, Bank, Telco, Mushroom)
- `CONFIG`: Experiment hyperparameters (seeds: 42, 123, 456, 789, 2024; context size: 10,000; K=50 clusters)
- `DEVICE`: Computation device (cuda/cpu)

## Reproducibility

All experiments use fixed random seeds (42, 123, 456, 789, 2024) as specified in `config.py`. Results are saved to the `results/` directory in JSON format. The main comparison and statistical tests use 5 seeds; sensitivity and orthogonality experiments use 3 seeds as documented in the paper.

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
