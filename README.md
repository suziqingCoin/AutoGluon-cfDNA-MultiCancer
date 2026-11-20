# AutoGluon-cfDNA-MultiCancer

Code and aggregate results for **Integrative Multi-Layer Cell-Free DNA Signatures Enable Multi-Cancer Early Detection from Shallow Sequencing**.

## Overview

This study analyzes published SPOT-MAS data from **2,288 participants**: 1,550 healthy controls and 136 lung, 223 breast, 159 colorectal, 122 liver, and 98 gastric cancer cases. The input comprises 8,630 processed features from six cfDNA families: end motifs (EM), copy-number aberrations (CNA), targeted-region methylation density (TMD), genome-wide methylation density (GWMD), genome-wide fragmentation profiles (GWFP), and fragment length (FLEN).

Four feature-ranking methods—Fscore, ANOVA, ReliefF, and Binomial—are combined with XGBoost-guided feature-count optimization and AutoGluon classification. Each task uses a stratified, patient-disjoint 80:20 training/test split and internal-validation model selection, without resampling or class weighting (`none`). Five binary tasks also undergo full-cohort Nested LOOCV, repeating feature and model selection within each outer training set.

## Main Results

The fixed-test models below were selected using internal validation. Nested LOOCV evaluates the selection procedure, with a separate model selected in each outer fold.

| Cancer | Feature ranking | Features | Fixed-test model | Fixed-test ROC AUC | Nested LOOCV ROC AUC |
|---|---|---:|---|---:|---:|
| [Lung](Lung_cancer/README.md) | ReliefF | 149 | WeightedEnsemble_L2 | 0.896749 | 0.930949 |
| [Breast](Breast_cancer/README.md) | ANOVA | 151 | WeightedEnsemble_L2 | 0.944158 | 0.955200 |
| [Colorectal](CRC_cancer/README.md) | ReliefF | 121 | NeuralNetFastAI | 0.987903 | 0.953090 |
| [Liver](Liver_cancer/README.md) | ReliefF | 19 | WeightedEnsemble_L2 | 0.952304 | 0.940904 |
| [Gastric](Gastric_cancer/README.md) | Fscore | 10 | WeightedEnsemble_L2 | 0.850806 | 0.895925 |

The separate [six-class analysis](pancancer/README.md) uses 20 ReliefF features and NeuralNetFastAI. On 458 held-out participants, weighted one-versus-rest AUC is **0.920262**, overall accuracy is **0.759825**, and healthy specificity is **0.964516**. Six-class Nested LOOCV was not performed.

## Repository Structure

```text
Lung_cancer/ ... Gastric_cancer/  Five binary analyses and task-specific results
pancancer/                      Six-class analysis and Figure 2E/F
feature_screen/                 Shared feature selection, training, and statistics
nested_loocv/                   Nested validation code and pooled results
fig2/                           Figure 2 plotting and supplied PDFs
environment/                    Dependency requirements and version records
docs/                           Reproduction guide and paper-to-code mapping
audit/                          Aggregate-result verification tools
```

## Installation

The recorded analysis environment uses **Python 3.11.9 and AutoGluon 1.1.1**. For public figures and result checks:

```bash
python -m pip install -r environment/requirements-figures.txt
```

See [environment setup](environment/README.md) for training and original-style plotting dependencies.

## Reproducing the Results

Run from the repository root:

```bash
python audit/validate_release.py
python fig2/prepare_fig2_inputs.py
python fig2/fig2.py
python Lung_cancer/code/plot_public.py
```

Each task provides `code/plot_public.py` for supplementary aggregate redraws. Figure 2 outputs go to `fig2/reproduced/`; supplementary outputs go to each task's `figures/public_reproduced/`. Supplied PDFs are preserved. Aggregate supplementary redraws use simplified styling; original-style plotting has additional input requirements described in the [supplementary guide](docs/SUPPLEMENTARY_FIGURES.md).

Model fitting requires the patient-level inputs described below. Use a [task README](Lung_cancer/README.md) for fixed-test commands, the [Nested LOOCV guide](nested_loocv/README.md) for full-cohort analysis, and the [reproduction guide](docs/REPRODUCIBILITY.md) for detailed instructions and [figure/table sources](docs/paper_artifacts.csv).

## Data Availability

The original SPOT-MAS cohort is described by [Nguyen et al., eLife (2023)](https://doi.org/10.7554/eLife.89083). Processed data are deposited as [ELIFE pan-cancer data, version 1](https://doi.org/10.6084/m9.figshare.24145392.v1), with a CC BY 4.0 license recorded in its DOI metadata.

The current analysis tree supplies selected features, aggregate metrics, ROC/PR coordinates, and confusion counts. Training requires the processed patient-level matrix; stage analyses also require matching clinical metadata. Sample-linked splits and prediction vectors are not bundled with the current analysis files. See [DATA_ACCESS.md](DATA_ACCESS.md) for input schemas, access conditions, and source-matching limits. Raw-sequencing preprocessing is outside this implementation.

## Reproducibility Notes

All evaluations use the same cohort and provide internal validation; no independent external validation is available. The fixed test set was examined in earlier experiments. Confidence intervals are conditional on fitted predictions and exclude full selection uncertainty and overlapping-fold dependence; stage-specific operating thresholds are retrospective. Full retraining, a clean installation of the full training environment, and complete patient-dependent figure replay have not been independently verified.

## Citation

Please cite **Integrative Multi-Layer Cell-Free DNA Signatures Enable Multi-Cancer Early Detection from Shallow Sequencing** when using this work. Publication details and a DOI will be added when available. Cite the original SPOT-MAS publication and dataset when using the source data.

## License

No license for this repository's code has been specified. Contact the repository authors for reuse permission. The upstream dataset's CC BY 4.0 license applies to that dataset and does not license this code.
