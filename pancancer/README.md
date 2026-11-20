# Six-Class Cancer Classification

## Overview

This analysis assigns the 2,288 participants to six classes: **Control/Healthy, Breast, CRC, Gastric, Liver, and Lung**, in that order. The stratified split contains 1,830 training participants and 458 fixed-test participants. Inputs are the same six cfDNA feature families used in the binary analyses.

## Feature Selection and Model

The fixed-test model is **NeuralNetFastAI with 20 ReliefF features**. The [shared pipeline](../feature_screen/README.md) ranks features on inner training, optimizes feature count with XGBoost, compares 20 method/model candidates on internal validation, and refits using the full training set. The six-class inner split uses seed 43. No resampling or class weighting is applied.

## Performance

| Fixed-test metric | Value |
|---|---:|
| Weighted OvR ROC AUC | 0.920262 |
| Weighted OvR ROC AUC, 95% CI | 0.896877–0.940055 |
| Macro OvR ROC AUC | 0.903266 |
| Weighted AUPRC (average precision) | 0.804223 |
| Overall six-class accuracy | 0.759825 |
| Healthy specificity | 0.964516 |
| Cancer detection recall | 0.635135 |
| Exact cancer-type recall among all cancer cases | 0.331081 |

Healthy specificity is the fraction of healthy participants assigned to Healthy (299/310). Cancer detection recall counts any cancer prediction among cancer cases (94/148); exact cancer-type recall requires the correct cancer label (49/148), including cases predicted as Healthy in the denominator. These definitions distinguish detection from cancer-type identification.

Sources: [final test metrics and CIs](results/tables/no_sampling_final_test_metrics_with_CI.csv), [macro/weighted summary](results/six_class_overall_macro_weighted_metrics.csv), and [six-class confusion counts](figures/data/Figure2F_confusion_counts.csv). [Per-class metrics](results/tables/no_sampling_per_class_metrics_with_CI.csv) use `dataset == reused_holdout_test` for the fixed test. **Six-class Nested LOOCV was not performed.**

## Code and Results

| File or directory | Purpose |
|---|---|
| [code/run_none.py](code/run_none.py) | Six-class entry calling the shared training implementation |
| [code/config.json](code/config.json) | Class order and reported model settings |
| [results/final_selected_features.csv](results/final_selected_features.csv) | The 20 selected features |
| [results/final_model_configuration.json](results/final_model_configuration.json) | Model configuration and parameter sources |
| [results/holdout/all_none_inner_validation_metrics.csv](results/holdout/all_none_inner_validation_metrics.csv) | Internal-validation model comparison |
| [code/plot_Figure2EF.py](code/plot_Figure2EF.py) | Figure 2E/F ROC and confusion matrix |
| [code/plot_public.py](code/plot_public.py) | S6 aggregate redraws |
| [code/private_replay/plot_s6.py](code/private_replay/plot_s6.py) | S6 original-style functions and input requirements |

## Usage

From the repository root after [installation](../environment/README.md):

```bash
# No patient data or fitting
python pancancer/code/run_none.py --check-config
python pancancer/code/plot_public.py
python fig2/prepare_fig2_inputs.py
python pancancer/code/plot_Figure2EF.py

# Patient-level matrix required; use a new empty output directory
python pancancer/code/run_none.py --data data/cohort.csv --output-dir private/none/pan6
```

Public figure outputs go to `pancancer/figures/public_reproduced/`. Use `python pancancer/code/private_replay/plot_s6.py --list` for original-style input requirements. See the [reproduction guide](../docs/REPRODUCIBILITY.md) for additional options.

## Notes

Figure 2E/F describes all six classes. Some S6 matrices display cancer classes only or cross-indicator accuracy; they are distinct from the full confusion matrix. The [plot manifest](figures/plot_manifest.json) specifies each output. Original patient-dependent plots require saved probabilities; aggregate redraws use simplified styling. Shared data and evaluation limits are in the [project README](../README.md#reproducibility-notes).
