# Reproducing the paper analyses

Use the [recorded environment](../environment/README.md). Public figures and checks need only aggregate files; model fitting needs the authorized inputs specified in [Data access](../DATA_ACCESS.md).

## Public figures and checks

From the repository root:

```bash
python audit/validate_release.py
python fig2/prepare_fig2_inputs.py
python fig2/fig2.py
python Lung_cancer/code/plot_public.py
```

Repeat the last command for Breast_cancer, CRC_cancer, Liver_cancer, Gastric_cancer, and pancancer. Figure 2 writes generated inputs and seven PDFs to ignored `fig2/reproduced/`. To use another directory:

```bash
python fig2/prepare_fig2_inputs.py --output-dir private/fig2
python fig2/fig2.py --input-dir private/fig2 --output-dir private/fig2
```

Supplementary redraws go to `figures/public_reproduced/` and use simplified styling. Original manuscript PDFs, semantic figure modules, and private replay entry points remain available. Exact patient-dependent replay needs the sample-linked inputs listed separately in each plot manifest; the CRC individual-patient score strip has no aggregate equivalent. Existing PDFs are not overwritten by public redraw commands. For original-style replay, use the S1–S6 modules in the [supplementary plotting guide](SUPPLEMENTARY_FIGURES.md); their styling hooks are scoped to each plot and do not run on import.

## Fixed-test model fitting

Each task has `code/run_none.py`; for example:

```bash
python Lung_cancer/code/run_none.py --check-config
python Lung_cancer/code/run_none.py --data data/cohort.csv --dry-run
python Lung_cancer/code/run_none.py --data data/cohort.csv --output-dir private/none/lung
```

The dry run validates schema, unique patients, labels, and original split hashes. Preserve input row/column order. New fits require an empty output directory and do not replace published results. Add `--training-cv` for the separate training-only three-fold comparison.

Summarize newly generated private predictions with:

```bash
python feature_screen/none/summarize_predictions.py --predictions private/none/lung/results/lung/holdout/none/selected_evaluation_predictions.csv --output-dir private/stats/lung
```

The [fixed-test protocol](../feature_screen/none/PROTOCOL.md) specifies ranking, K search, selection, and refit rules. The inner-training feature prefix is retained for full-training refitting. The model is selected using internal AUC/AP/recall/specificity and deterministic tie-breaks. Test candidate comparisons are descriptive. Liver uses the reported `none` ReliefF/K=19 result.

## Nested LOOCV

See the [Nested LOOCV guide](../nested_loocv/README.md) and [protocol](../nested_loocv/PROTOCOL.md). Use a fresh private output tree; completed checkpoints have source-specific signatures and cannot be resumed with modified source. A full run is expensive and is not needed to redraw completed results.

## Canonical result sources

[paper_artifacts.csv](paper_artifacts.csv) maps paper items to the retained sources and code. The table labels below identify manuscript outputs; the canonical CSVs live alongside the corresponding analyses:

| Paper table | Canonical source |
|---|---|
| S3 | Five binary `results/tables/all_no_sampling_test_candidate_metrics.csv` files, concatenated in Lung/Breast/CRC/Liver/Gastric order |
| S4–S6 | `nested_loocv/results/Table_S4.csv` through `Table_S6.csv` |
| S7 | `pancancer/results/tables/no_sampling_per_class_metrics_with_CI.csv`, filtered to `dataset == reused_holdout_test` |

Training-CV summaries remain separate from full-cohort Nested LOOCV. Final model records retain all 20 internal-validation candidates, each method's feature/K search, the chosen method's trained parameters/refit evidence, split hashes, and final metrics. Historical notebooks, duplicate reports, and nonfinal intermediate records are kept in a separate internal archive.

## Interpretation and verification scope

All analyses use the same cohort; historical fixed-test reuse prevents interpreting it as new external validation. CIs use 2,000 class-stratified bootstrap replicates conditional on fitted predictions. They omit full procedure uncertainty and overlapping-fold dependence. Retrospective specificity thresholds are exploratory; stage missingness is retained and stage AUC is suppressed for n < 10.

Public checks can verify aggregate consistency and source hashes, but cannot independently verify patient-level predictions without approved access. [Prior private-source checks](../audit/private_source_validation.json) and [source algorithm comparisons](../audit/algorithm_equivalence.json) retain that evidence and its scope. They do not imply training was rerun during this cleanup. Clean installation of the full training environment, full retraining, exact patient-dependent plotting replay, and cohort-independent discovery of targeted methylation regions remain unverified. Raw sequencing-to-feature extraction is outside the supplied code.
