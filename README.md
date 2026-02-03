# Risk-Aware Adaptive Inference for Tree Ensembles

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Tree ensembles are widely deployed for tabular prediction, but their inference cost grows linearly with the number of trees or boosting stages, which can violate latency and throughput budgets in production systems. Adaptive inference (early exiting) reduces expected evaluation work by stopping on "easy" inputs, yet common confidence heuristics are often miscalibrated and can fail under distribution shift, causing premature exits with elevated error among exited samples.

We propose **Risk-Aware Adaptive Inference (RAAI)**, a deployment-oriented framework that replaces heuristic stopping with a calibrated *flip-risk* signal: the probability that a prefix (partial-ensemble) prediction would disagree with the full-ensemble prediction. RAAI then applies a closed-loop, OOD-aware stopping policy that tightens admissible flip-risk and can disable early exit for distributionally suspicious inputs, mitigating the principal failure mode of early exit under shift.

## Key Features

- **Calibrated Flip-Risk Estimation**: Learn a lightweight calibrator to estimate the probability that a prefix prediction agrees with the full-ensemble prediction
- **OOD-Aware Stopping Policy**: Closed-loop controller that conditions stopping tolerance on an OOD suspicion score
- **Two-Level Suspicious Policy with Hard Gate**: Conservative risk allocation for suspicious inputs with optional full-inference escalation
- **Support for RF and GBM**: Works with both Random Forest and Gradient Boosting ensembles

## Results

RAAI achieves substantial in-distribution work reduction (63–84% fewer evaluated components) while maintaining low disagreement with the full ensemble, and reduces early-exit failure rates on near- and far-OOD shifts.

| Dataset | Model | Work Reduction (%) | Disagreement Rate (%) | Accuracy (%) |
|---------|-------|-------------------|----------------------|--------------|
| MNIST | RF | 77.0 ± 0.2 | 1.0 ± 0.1 | 95.5 ± 0.2 |
| MNIST | GBM | 73.3 ± 0.4 | 1.2 ± 0.1 | 92.4 ± 0.2 |
| Covertype | RF | 84.1 ± 0.3 | 2.0 ± 0.4 | 86.2 ± 0.5 |
| Covertype | GBM | 83.3 ± 0.7 | 2.8 ± 0.2 | 79.1 ± 0.5 |
| HIGGS | RF | 63.0 ± 0.6 | 3.2 ± 0.2 | 70.1 ± 0.7 |
| HIGGS | GBM | 66.2 ± 1.3 | 2.3 ± 0.3 | 69.7 ± 0.7 |

## Installation

### Requirements

- Python 3.8+
- NumPy
- scikit-learn
- SciPy
- matplotlib (for visualization)
- joblib

### Setup

```bash
# Clone the repository
git clone https://github.com/pz1004/risk-aware_adaptive_inference_for_tree_ensembles.git
cd risk-aware_adaptive_inference_for_tree_ensembles

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scikit-learn scipy matplotlib joblib
```

### Data Preparation

Download the required datasets and place them in the `data/` directory:

1. **MNIST**: Download from [https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)
   - Place files in `data/mnist/MNIST/raw/`

2. **Covertype**: Download from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/covertype)
   - Place processed files in `data/covertype/`

3. **HIGGS**: Download from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS)
   - Place the CSV file in `data/higgs/`

## Usage

### Running the Proposed Method

```bash
cd src
python run_proposed.py --dataset mnist --model rf
```

#### Command-line Arguments

- `--dataset`: Dataset name (`mnist`, `covertype`, or `higgs`)
- `--model`: Model type (`rf` for Random Forest, `gbm` for Gradient Boosting)
- `--n_runs`: Number of experimental runs (default: 3)
- `--delta_id`: Risk threshold for ID samples (default: 0.05)
- `--delta_suspicious`: Risk threshold for suspicious samples (default: 0.01)
- `--hard_gate_threshold`: Threshold for forcing full inference (default: 0.8)

### Running Baselines

```bash
cd src
python run_baselines.py --dataset mnist --model rf
```

### Running Ablation Studies

```bash
cd src
python run_ablation.py --dataset mnist --model rf --ablation_type hard_gate
```

Ablation types:
- `hard_gate`: Sensitivity to hard gate threshold
- `delta_id`: Sensitivity to ID risk threshold
- `delta_suspicious`: Sensitivity to suspicious risk threshold

### Running All Experiments

```bash
cd src
python run_final_experiments.py
```

## Project Structure

```
.
├── README.md
├── src/
│   ├── baselines.py        # OOD detection baselines (MSP, Entropy, IForest, etc.)
│   ├── calibration.py      # Flip-risk calibration module
│   ├── data_loaders.py     # Dataset loading and preprocessing
│   ├── models.py           # RF/GBM training and prefix evaluation
│   ├── run_ablation.py     # Ablation study experiments
│   ├── run_baselines.py    # Baseline comparisons
│   ├── run_proposed.py     # Main RAAI algorithm
│   ├── stopping_policy.py  # Risk-aware stopping policy
│   └── suspicion.py        # OOD suspicion models (KNN, Trajectory)
├── baselines/              # Baseline results
├── results/                # Experiment results
└── data/                   # Datasets (not included, see Data Preparation)
```

## Algorithm Overview

### RAAI Inference Procedure

```
Input: Ensemble F={f_1,...,f_T}, input x, calibrator C, KNN index I
Output: Prediction ŷ and stopping time t*

1. Compute OOD suspicion score u(x) via KNN distance
2. If u(x) > τ_gate: return full inference (ŷ_T, T)
3. Set δ and t_min based on u(x):
   - If u(x) ≤ τ_susp: δ = δ_ID, t_min = t_min^ID
   - Else: δ = δ_susp, t_min = t_min^susp
4. For t = 1 to T:
   a. Evaluate component f_t and update state
   b. If t < t_min: continue
   c. Compute prefix prediction ŷ_t and features φ_t
   d. Estimate p̂_agree = C(φ_t)
   e. If p̂_agree ≥ 1-δ: return (ŷ_t, t)
5. Return (ŷ_T, T)
```

## Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_estimators` | Number of trees/stages | 100 |
| `k` (KNN) | Number of neighbors for suspicion | 10 |
| `τ_susp` | Suspicion threshold | 0.5 |
| `τ_gate` | Hard gate threshold | 0.7-0.9 |
| `δ_ID` | Risk tolerance for ID samples | 0.03-0.10 |
| `δ_susp` | Risk tolerance for suspicious samples | 0.001-0.05 |
| `t_min^ID / T` | Minimum depth fraction (ID) | 0.1 |
| `t_min^susp / T` | Minimum depth fraction (suspicious) | 0.3 |
