# Nested Leave-One-Out Cross-Validation

## Overview

Nested LOOCV evaluates the complete feature and model selection procedure for five cancer-versus-control tasks. Every participant in a task receives one prediction from an outer model trained without that participant. All tasks use no resampling or class weighting.

The completed analyses comprise **8,488 outer task–participant folds**. Controls are reused across tasks, so this is not a count of 8,488 independent participants. No six-class Nested LOOCV is reported.

## Validation Workflow

1. Hold out one participant as the outer test case.
2. Split the remaining patients into inner training and validation sets; fit fresh Fscore, ANOVA, ReliefF, and Binomial rankings on inner training only.
3. Optimize feature count with XGBoost for each ranking, then compare five AutoGluon model families per method.
4. Lock method, feature count, and model using internal-validation metrics.
5. Rerank all outer-training patients with the selected method, fit fresh preprocessing, and refit the selected base-model templates on outer training. Ensemble weights remain inner-validation-fitted.
6. Predict the held-out participant once, then pool predictions across completed outer folds.

No supervised ranking is shared across outer folds. The [protocol](PROTOCOL.md) gives the detailed selection, preprocessing, and version-dependent refit rules.

## Main Results

| Task | Outer folds | Pooled ROC AUC | 95% CI | AUPRC (average precision) |
|---|---:|---:|---|---:|
| Lung | 1686 | 0.930949 | 0.909064–0.950043 | 0.707779 |
| Breast | 1773 | 0.955200 | 0.942082–0.966975 | 0.815189 |
| Colorectal | 1709 | 0.953090 | 0.927344–0.973746 | 0.878295 |
| Liver | 1672 | 0.940904 | 0.923040–0.957398 | 0.603186 |
| Gastric | 1648 | 0.895925 | 0.864997–0.923805 | 0.531976 |

Values are calculated from pooled out-of-fold predictions, not averages of single-patient fold metrics. [Table S4](results/Table_S4.csv) contains the full metric set.

## Code Structure

| Location | Purpose |
|---|---|
| [scripts/check_config.py](scripts/check_config.py) | Check configuration and imports without training |
| [scripts/run.py](scripts/run.py) | Schedule the five tasks and their outer folds |
| [scripts/run_fold.py](scripts/run_fold.py) | Run an individual outer fold |
| [src/protocol.py](src/protocol.py) | Task definitions, paths, selection settings, and K search |
| [src/feature_rankings.py](src/feature_rankings.py) | Training-only feature ranking |
| [src/fold.py](src/fold.py) | Inner selection, outer-training refit, and fold validation |
| [src/report.py](src/report.py) | Pooled metrics, confidence intervals, and stage summaries |
| [results/](results/) | Completed aggregate reports and per-task summaries |

## Running the Analysis

From the repository root after [training environment setup](../environment/README.md):

```bash
# No patient data or training
python nested_loocv/scripts/check_config.py

# New full analysis: requires patient-level molecular and clinical inputs
export CFDNA_RAW="$PWD/data/cohort.csv"
export CFDNA_CLINICAL="$PWD/data/clinical.xlsx"
export CFDNA_NESTED_OUTPUT="$PWD/private/nested_new"
python nested_loocv/scripts/run.py
python nested_loocv/src/report.py
```

Use a fresh output directory under ignored `private/` or outside the repository. The scheduler, single-fold entry point, and report command reject output locations elsewhere in the repository to protect published results. For a single fold, replace the scheduler command with `python nested_loocv/scripts/run_fold.py lung 0`; complete reporting still requires all folds for the task. See [DATA_ACCESS.md](../DATA_ACCESS.md) for exact inputs. Re-running the full analysis is unnecessary for plotting existing aggregate results.

## Outputs

- [Table S4](results/Table_S4.csv): pooled overall performance.
- [Table S5](results/Table_S5.csv): stage I–II sensitivity at retrospective 95%/97% specificity work points.
- [Table S6](results/Table_S6.csv): stage-specific results.
- [Figure 2C](results/Figure_2C.csv) and [Figure 2D](results/Figure_2D.csv): direct plotting sources.
- `results/<task>/`: task-level metrics, stage missingness, method/model selection frequencies, and feature-count summaries.

The [Figure 2 renderer](../fig2/README.md) uses these public aggregates without patient-level data.

## Notes

A complete run is computationally expensive and needs the processed patient matrix; clinical metadata are required for stage reporting. Confidence intervals use 2,000 patient-level class-stratified bootstrap replicates conditional on fitted predictions, excluding full procedure uncertainty and dependence from overlapping training folds. Specificity thresholds are retrospective, stage missingness is preserved, and stage AUC is suppressed for groups smaller than 10. These are internal evaluations of the same cohort, without independent external validation.
