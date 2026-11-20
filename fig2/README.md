# Figure 2

Figure 2 combines selected-feature composition, five-cancer performance, early-stage detection, and six-class classification. The supplied PDFs are preserved; public redraws use the aggregate sources below.

## Panels and Data Sources

All source paths are relative to the repository root.

| Panel | Content | Source |
|---|---|---|
| A | Feature-family composition of the five selected binary panels | Each binary task's `results/final_selected_features.csv` |
| B | Fixed-test binary ROC AUC and 95% CI | Each binary task's `results/tables/no_sampling_final_test_metrics_with_CI.csv` |
| C | Five-cancer Nested LOOCV ROC AUC | [nested_loocv/results/Figure_2C.csv](../nested_loocv/results/Figure_2C.csv) |
| D | Stage I–II sensitivity at retrospective 95% specificity | [nested_loocv/results/Figure_2D.csv](../nested_loocv/results/Figure_2D.csv) |
| E | Six-class fixed-test ROC curves | [ROC coordinates](../pancancer/figures/data/Figure2E_ROC_coordinates.csv) and [AUC values](../pancancer/figures/data/Figure2E_ROC_AUC.csv) |
| F | Six-class confusion matrix for 458 test participants | [Confusion counts](../pancancer/figures/data/Figure2F_confusion_counts.csv) and normalized counts in the same directory |

## Reproduce the Figure

From the repository root after [installing figure dependencies](../environment/README.md):

```bash
python fig2/prepare_fig2_inputs.py
python fig2/fig2.py
```

[prepare_fig2_inputs.py](prepare_fig2_inputs.py) creates eight TXT inputs and a source-hash manifest. [fig2.py](fig2.py) validates these inputs and writes six panel PDFs, a combined PDF, and a validation report to `fig2/reproduced/`. Neither command requires patient-level data or model training.

To choose a separate destination:

```bash
python fig2/prepare_fig2_inputs.py --output-dir private/fig2
python fig2/fig2.py --input-dir private/fig2 --output-dir private/fig2
```

Single panels are 93 × 56.667 mm; the combined figure is 186 × 170 mm. Arial is used when available, with a recorded fallback otherwise. See the [reproduction guide](../docs/REPRODUCIBILITY.md) for supplementary figures and the [project notes](../README.md#reproducibility-notes) for statistical interpretation.
