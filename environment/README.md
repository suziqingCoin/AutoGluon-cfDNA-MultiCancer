# Software Environment

The recorded analysis environment uses **Python 3.11.9 and AutoGluon 1.1.1** on Linux with CPU execution. Use the recorded AutoGluon version for training because the refit APIs are version dependent.

## Dependency Files

| File | Purpose |
|---|---|
| [requirements-figures.txt](requirements-figures.txt) | Public aggregate plotting and result validation |
| [requirements-training.txt](requirements-training.txt) | Model training, feature selection, and Nested LOOCV |
| [requirements-private-replay.txt](requirements-private-replay.txt) | Additional original-style plotting dependencies, including seaborn |
| [pip_freeze_original.txt](pip_freeze_original.txt) | Full installed-package version record from the analysis environment |

## Installation

Run from the repository root. For public figures and validation:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment/requirements-figures.txt
```

The public figure requirements have been installed and the aggregate validator has passed in a fresh Python 3.11.9 environment. This check does not cover the training dependencies.

For model fitting, install the training requirements in the intended training environment:

```bash
python -m pip install -r environment/requirements-training.txt
```

For original-style supplementary plotting, also install:

```bash
python -m pip install -r environment/requirements-private-replay.txt
```

The recorded Torch version is `2.3.1+cpu`; resolving that build may require the appropriate CPU wheel index. A clean installation of the full recorded environment has not been verified. Model training and complete Nested LOOCV require substantial CPU time and memory.

Arial is separately licensed and is not bundled. Plotters use a fallback when it is unavailable, so typography may differ. Installing dependencies does not provide the patient-level inputs needed for model fitting or patient-dependent figure replay; see [DATA_ACCESS.md](../DATA_ACCESS.md).
