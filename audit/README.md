# Result Verification

The public validator checks consistency between reported models, aggregate results, figure inputs, and analysis sources without fitting models or reading patient-level data.

## Usage

From the repository root with the [figure dependencies](../environment/README.md) installed:

```bash
python audit/validate_release.py
```

[validate_release.py](validate_release.py) checks the six internally selected model configurations, 120 candidate test summaries, selected features, split-hash records, refit metadata, and 8,488 completed Nested LOOCV fold counts. It also checks canonical table sources, Figure 2 inputs, plotting-function mappings, local Markdown links, Python syntax, file hashes, and basic file hygiene. The report is written to ignored `audit/public_validation.json`.

## Supporting Records

| File | Purpose |
|---|---|
| [release_manifest.csv](release_manifest.csv) | File sizes and SHA256 checksums for the documented source version |
| [private_source_validation.json](private_source_validation.json) | Recorded metric/checksum comparisons against authorized source artifacts |
| [algorithm_equivalence.json](algorithm_equivalence.json) | Recorded comparisons of scientific implementations |

The checksum manifest must match the source and documentation version being checked; documentation changes require a corresponding manifest update before the complete hash check can pass. The supporting records describe previous checks, not fresh training or independent patient-level replication.

After reviewing intentional changes, refresh the inventory and validate it:

```bash
python audit/update_manifest.py
python audit/validate_release.py
```

The GitHub Actions workflow runs aggregate validation and public redraws on pushes and pull requests. It does not refresh the manifest automatically.

Public aggregate checks do not verify full model retraining, data-access authorization, or all sensitive content in Git history. See the [reproduction guide](../docs/REPRODUCIBILITY.md) for the scope of result reproduction.
