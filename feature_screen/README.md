# Shared Feature Selection and Model Training Pipeline

This directory implements the fixed-test workflow for all five cancer-versus-control tasks and the six-class task. It includes feature selection, model fitting, refitting, prediction export, and evaluation, using no resampling or class weighting (`none`).

## Workflow

1. Split original patients into stratified 80% training and 20% test sets; form a further training-only holdout for internal validation.
2. Rank features on inner-training patients using Fscore, ANOVA, ReliefF, or Binomial scoring.
3. Optimize each ranking's prefix length with incremental XGBoost fits and validation AUC, subject to task-specific caps and stopping rules.
4. Compare XGBoost, RandomForest, LightGBM, NeuralNetFastAI, and WeightedEnsemble_L2 for each optimized panel: 20 candidates per task.
5. Select by internal-validation AUC, average precision, recall, and healthy specificity, followed by deterministic tie-breaks. Refit using the full training set while retaining the chosen inner-training feature prefix; ensemble weights remain validation-fitted.
6. Evaluate the locked choice on the test set. Candidate test comparisons are descriptive and do not determine selection.

The [protocol](none/PROTOCOL.md) specifies split seeds, K-search rules, selection order, and AutoGluon refit behavior. Full-cohort [Nested LOOCV](../nested_loocv/README.md) is implemented separately.

## Code Structure

Each task has a small CLI entry and its own configuration; the analysis implementation is shared:

```text
Breast_cancer/code/run_none.py
    -> feature_screen/none/runner.py:main(config)
    -> feature_screen/none/none_experiment.py:context_run(...)
```

With `--training-cv`, the runner calls `run_task(...)` for the fixed-test analysis and an additional three-fold comparison within the training set.

| Module | Role |
|---|---|
| [none/runner.py](none/runner.py) | CLI options, matrix/schema checks, original split-hash verification, and output handling |
| [none/feature_rankings.py](none/feature_rankings.py) | `rank(...)`: four feature-ranking methods, including vectorized ReliefF |
| [none/none_protocol.py](none/none_protocol.py) | Shared settings, feature-count caps, and utilities |
| [none/none_experiment.py](none/none_experiment.py) | `optimization`, `choose`, `exact_refit`, `context_run`, and point metrics in `score` |
| [none/patient_statistics.py](none/patient_statistics.py) | Patient-stratified bootstrap metrics and exact binomial intervals |
| [none/summarize_predictions.py](none/summarize_predictions.py) | Metrics and CIs from saved patient-level predictions |
| [plot_public.py](plot_public.py) | Supplementary aggregate plotting shared by task entry points |
| [original_plots.py](original_plots.py) | Input adapters and plotting context for original-style figures |

## Running the Pipeline

From the repository root, after [environment setup](../environment/README.md):

```bash
# Configuration and imports only; no patient data or training
python Breast_cancer/code/run_none.py --check-config

# With the required patient-level matrix; no fitting
python Breast_cancer/code/run_none.py --data data/cohort.csv --dry-run

# New analysis; requires an empty output directory
python Breast_cancer/code/run_none.py --data data/cohort.csv --output-dir private/none/breast
```

The other five task entries use the same arguments. Input row/column order must match the [data specification](../DATA_ACCESS.md). New results and patient-level predictions are written under the chosen private output directory.

## Statistical Summaries

Point metrics include ROC AUC, AUPRC (average precision), sensitivity, healthy specificity, accuracy, precision, and F1. Six-class AUC/AP additionally use weighted and macro OvR summaries. The prediction summarizer computes 2,000 class-stratified bootstrap replicates and exact healthy-specificity intervals:

```bash
# Requires predictions from an authorized analysis run
python feature_screen/none/summarize_predictions.py --predictions private/none/breast/results/breast/holdout/none/selected_evaluation_predictions.csv --output-dir private/stats/breast
```

These intervals are conditional on fitted predictions; they do not repeat model training or selection. See the [reproduction guide](../docs/REPRODUCIBILITY.md) for result locations and interpretation.
