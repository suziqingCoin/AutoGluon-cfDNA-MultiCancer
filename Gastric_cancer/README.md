# Gastric Cancer Classification

## Overview

This task distinguishes gastric cancer from healthy controls using the six cfDNA feature families in the [study overview](../README.md). It includes **1,648 participants** (98 cancer cases and 1,550 controls), with 1,318 assigned to training and 330 to the fixed test set.

## Feature Selection and Model

The fixed-test analysis selects **10 Fscore features** and **WeightedEnsemble_L2**. Feature ranking uses inner-training patients; XGBoost optimizes the feature count, and internal validation selects among four ranking methods and five AutoGluon model families. The chosen feature prefix and model are refitted on the full training set before test evaluation, without resampling or class weighting.

The task entry calls the [shared training pipeline](../feature_screen/README.md). Nested LOOCV repeats selection in each outer fold, so its feature panels and models can differ from the fixed-test choice.

## Performance

| Metric | Fixed test (n = 330) | Nested LOOCV (n = 1648) |
|---|---:|---:|
| ROC AUC | 0.850806 | 0.895925 |
| ROC AUC, 95% CI | 0.745153–0.942258 | 0.864997–0.923805 |
| AUPRC (average precision) | 0.570471 | 0.531976 |
| Sensitivity | 0.400000 | 0.326531 |
| Healthy specificity | 0.990323 | 0.990968 |

Sources: [fixed-test metrics](results/tables/no_sampling_final_test_metrics_with_CI.csv) and [pooled Nested LOOCV metrics](../nested_loocv/results/gastric/Table_S4.csv). Both are internal evaluations of the same cohort; shared interpretation limits are summarized in the [project README](../README.md#reproducibility-notes).

## Code and Results

| File or directory | Purpose |
|---|---|
| [code/run_none.py](code/run_none.py) | Task CLI calling the shared training implementation |
| [code/config.json](code/config.json) | Task settings and reported fixed-test model |
| [results/final_selected_features.csv](results/final_selected_features.csv) | Selected features and ranking scores |
| [results/final_model_configuration.json](results/final_model_configuration.json) | Model configuration and linked fitted parameters |
| [results/holdout/all_none_inner_validation_metrics.csv](results/holdout/all_none_inner_validation_metrics.csv) | The 20 candidates used for model selection |
| [results/tables/](results/tables/) | Test metrics, confidence intervals, and separate training-CV summaries |
| [code/plot_public.py](code/plot_public.py) | Supplementary plots from public aggregates |
| [code/private_replay/plot_s5.py](code/private_replay/plot_s5.py) | S5 original-style functions; input requirements vary by plot |

Figure 2A/B uses this task's selected features and fixed-test metrics; Figure 2C/D uses the [Nested LOOCV results](../nested_loocv/README.md). The [plot manifest](figures/plot_manifest.json) lists individual supplementary outputs and their inputs.

## Usage

From the repository root after [installation](../environment/README.md):

```bash
# No patient data or fitting
python Gastric_cancer/code/run_none.py --check-config
python Gastric_cancer/code/plot_public.py

# Patient-level matrix required; use a new empty output directory
python Gastric_cancer/code/run_none.py --data data/cohort.csv --output-dir private/none/gastric
```

The public renderer writes simplified redraws to `Gastric_cancer/figures/public_reproduced/`. Use `python Gastric_cancer/code/private_replay/plot_s5.py --list` to inspect original-style plot inputs. See [data requirements](../DATA_ACCESS.md) and the [reproduction guide](../docs/REPRODUCIBILITY.md) for split checks and optional training-only CV.

## Notes

The S5 module contains feature-count optimization, top-feature scores, and training/test metric heatmaps. This task includes 98 cancer cases; use the reported confidence intervals when interpreting performance.
