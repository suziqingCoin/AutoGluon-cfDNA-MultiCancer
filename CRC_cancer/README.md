# Colorectal Cancer Classification

## Overview

This task distinguishes colorectal cancer from healthy controls using the six cfDNA feature families in the [study overview](../README.md). It includes **1,709 participants** (159 cancer cases and 1,550 controls), with 1,367 assigned to training and 342 to the fixed test set.

## Feature Selection and Model

The fixed-test analysis selects **121 ReliefF features** and **NeuralNetFastAI**. Feature ranking uses inner-training patients; XGBoost optimizes the feature count, and internal validation selects among four ranking methods and five AutoGluon model families. The chosen feature prefix and model are refitted on the full training set before test evaluation, without resampling or class weighting.

The task entry calls the [shared training pipeline](../feature_screen/README.md). Nested LOOCV repeats selection in each outer fold, so its feature panels and models can differ from the fixed-test choice.

## Performance

| Metric | Fixed test (n = 342) | Nested LOOCV (n = 1709) |
|---|---:|---:|
| ROC AUC | 0.987903 | 0.953090 |
| ROC AUC, 95% CI | 0.970559–0.998488 | 0.927344–0.973746 |
| AUPRC (average precision) | 0.942857 | 0.878295 |
| Sensitivity | 0.687500 | 0.691824 |
| Healthy specificity | 1.000000 | 0.996774 |

Sources: [fixed-test metrics](results/tables/no_sampling_final_test_metrics_with_CI.csv) and [pooled Nested LOOCV metrics](../nested_loocv/results/crc/Table_S4.csv). Both are internal evaluations of the same cohort; shared interpretation limits are summarized in the [project README](../README.md#reproducibility-notes).

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
| [code/private_replay/plot_s3.py](code/private_replay/plot_s3.py) | S3 original-style functions; input requirements vary by plot |

Figure 2A/B uses this task's selected features and fixed-test metrics; Figure 2C/D uses the [Nested LOOCV results](../nested_loocv/README.md). The [plot manifest](figures/plot_manifest.json) lists individual supplementary outputs and their inputs.

## Usage

From the repository root after [installation](../environment/README.md):

```bash
# No patient data or fitting
python CRC_cancer/code/run_none.py --check-config
python CRC_cancer/code/plot_public.py

# Patient-level matrix required; use a new empty output directory
python CRC_cancer/code/run_none.py --data data/cohort.csv --output-dir private/none/crc
```

The public renderer writes simplified redraws to `CRC_cancer/figures/public_reproduced/`. Use `python CRC_cancer/code/private_replay/plot_s3.py --list` to inspect original-style plot inputs. See [data requirements](../DATA_ACCESS.md) and the [reproduction guide](../docs/REPRODUCIBILITY.md) for split checks and optional training-only CV.

## Notes

Figure S3C (`ClassifierScore_ByModel`) shows individual patient scores. Its original logic is retained, but the public renderer skips it because patient-level probabilities are required; it is not replaced with an aggregate plot.
