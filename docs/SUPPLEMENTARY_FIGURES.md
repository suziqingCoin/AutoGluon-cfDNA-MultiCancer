# Supplementary figure plotting

The six original-style modules group related panels by supplementary figure. Each exposes named plotting functions with an explicit `ReplayContext`; repeated train/test or method variants share a function while retaining their original labels, colors, axes, sizes, and save settings.

| Figure | Task | Original-style module |
|---|---|---|
| S1 | Lung | [plot_s1.py](../Lung_cancer/code/private_replay/plot_s1.py) |
| S2 | Breast | [plot_s2.py](../Breast_cancer/code/private_replay/plot_s2.py) |
| S3 | CRC | [plot_s3.py](../CRC_cancer/code/private_replay/plot_s3.py) |
| S4 | Liver | [plot_s4.py](../Liver_cancer/code/private_replay/plot_s4.py) |
| S5 | Gastric | [plot_s5.py](../Gastric_cancer/code/private_replay/plot_s5.py) |
| S6 | Six-class | [plot_s6.py](../pancancer/code/private_replay/plot_s6.py) |

Modules are grouped by supplementary figure. CRC score strips are Figure S3C. Other final panel letters are not assigned without a verified manuscript panel map; filenames and function mappings are explicit in each `figures/plot_manifest.json` and [paper_artifacts.csv](paper_artifacts.csv).

## Public aggregate redraws

Each task retains `code/plot_public.py`. It uses aggregate CSVs and simplified styling, and writes to `figures/public_reproduced/`. These redraws are not represented as exact original-style replay. The CRC patient score strip is skipped rather than replaced by another metric.

## Original-style replay

From the repository root:

```bash
python Breast_cancer/code/private_replay/plot_s2.py --list
# Original feature-family plot: public feature input is sufficient
python Breast_cancer/code/private_replay/plot_s2.py --only Breast-ANOVA-Barplot-BiomarkerTypes_NoGrid
# All groups: requires the authorized patient inputs listed by --list
python Breast_cancer/code/private_replay/plot_s2.py
```

The default output is ignored `private/replay/<task-directory>/`. `--output-dir` selects another separate destination. The original manuscript PDFs and public source manifests are not overwritten. Selecting a model ROC group emits its five model plots together.

Feature composition, feature scores, and K-search plots read only the provided aggregates. Patient-dependent AUC/ROC/metric/score plots require the frozen patient split, test probabilities, and apparent-training probabilities listed in the manifest. The lung PR plot uses the saved test probabilities directly. No models are fitted, no patients are resplit, and no test result is used to select a new model.

The shared [original_plots.py](../feature_screen/original_plots.py) provides task-scoped prediction adapters and an original-style save context. Arial regular, PDF/300-dpi PNG, crop/padding, random seed, and per-plot settings match the prior replay renderer. Global font/save/RNG hooks are restored after each figure; importing a module does not patch Matplotlib. Exact private replay additionally needs the [private replay dependencies](../environment/README.md).

CRC Figure S3C retains the true model-score strip logic, train/test panels, and seeded jitter. Its patient probabilities are not distributed. Missing private inputs produce an explicit error; no synthetic scores or aggregate substitute are used. Data access is described in [DATA_ACCESS.md](../DATA_ACCESS.md).
