# Data availability and access

The study comprises 2,288 participants: 1,550 controls and 136 lung, 223 breast, 159 colorectal, 122 liver, and 98 gastric cancer patients. The five binary analyses reuse the same controls; the six-class analysis uses the same cohort.

## Public materials

The repository includes aggregate performance tables, ROC/PR coordinates, confusion counts, selected molecular features, and analysis code. Public curve tables omit per-score probability thresholds; reported retrospective stage work-point thresholds remain in the aggregate stage results. Patient-level molecular and clinical records, sample-linked splits, and test/OOF prediction vectors are not distributed in this GitHub release; the upstream source is described below.

## Upstream data source

The original cohort publication, [Nguyen et al., eLife (2023)](https://doi.org/10.7554/eLife.89083), cites [ELIFE pan-cancer data, version 1](https://doi.org/10.6084/m9.figshare.24145392.v1) as its deposited data. The [DataCite DOI record](https://api.datacite.org/dois/10.6084/m9.figshare.24145392.v1) identifies Chi Nguyen as creator, describes metadata and processed data with 8,630 features, and records a **CC BY 4.0** dataset license. This dataset license does not establish a license for this repository's analysis code.

A file-by-file checksum match between the deposited files and the exact local analysis matrix has not been established. Use the versioned source DOI and follow its attribution terms. The dataset is not a raw-sequencing archive and does not provide an independently verified sequencing-to-feature reproduction pipeline here.

## Access and redistribution scope

This GitHub release does not redistribute patient-level molecular or clinical tables, sample-linked splits, or derived prediction vectors. This release choice should not be read as a claim that the upstream processed dataset is unavailable publicly. The source publication states that individual sequencing data require a request and data-sharing agreement with the original investigators; use its data-availability statement for that route.

For the exact processed/clinical inputs used by this statistical revision, contact the repository maintainers. For upstream data requests, use the correspondence details in the source publication. Please keep patient information out of public issues.

The specifications below describe the inputs needed for model fitting and report reconstruction. Public aggregate checks and redraws require neither the molecular matrix nor the clinical workbook.

## Processed molecular matrix

Provide the matrix as `data/cohort.csv` (ignored by Git), or set `CFDNA_RAW` / use the holdout `--data` flag. Exact first four columns in order: `SampleID,Cohort,Label,Diagnosis`; subsequent **8,630 molecular columns in original order**. Each SampleID occurs once, molecular values are finite. `Label=0` for Control and 1 for cancer; `Diagnosis` is `Control`, `Breast cancer`, `CRC cancer`, `Gastric cancer`, `Liver cancer`, or `Lung cancer`. Six-class integer targets are built from Diagnosis in that order. Preserve row order to reproduce exact stratified splits/fold indices; preserve column order for stable ranking ties. The committed split hashes and original raw SHA256 in nested integrity files identify the original dataset without exposing it. Edited/reordered data are not the same exact reproduction input.

## Clinical input for stage reporting

Rebuilding stage-specific Nested LOOCV summaries with [report.py](nested_loocv/src/report.py) requires a matching clinical workbook. Place it at `data/clinical.xlsx` or set `CFDNA_CLINICAL` to its path. The reader uses sheet `Table S1`, skips the first three rows, and reads the first 11 columns as Group, Label (diagnosis text), Source, SampleID, Age, Gender, Tumor Stage, Tumor type, Metastasis, Number of tumor, and Tumor diameter. 2,288 unique patient IDs are joined exactly. Stages are normalized by `stage()` in that module; missing/unknown values remain so, not imputed. This derived input workbook is not distributed with the code; obtaining upstream clinical metadata does not by itself verify that it matches this exact workbook.

## Generated predictions

Prediction columns: SampleID, Diagnosis, y_true, y_pred, p_0...p_1 (binary) or p_0...p_5 (six-class), task/context/strategy/method/model; Nested LOOCV adds fold and k. These outputs go under ignored `private/` or a user-chosen external output directory. They are deliberately not bundled. Full-fold report reconstruction additionally needs the newly generated or original authorized checkpoint artifacts; aggregate-only validation does not.

## Preprocessing scope

The code starts from the processed cfDNA feature matrix. Raw-sequencing-to-feature extraction, exact genome bin construction and clinical data acquisition are outside the supplied verified implementation; no end-to-end sequencing reproduction is claimed. The six family labels follow original feature identifiers: EM=`MOTIF_END*` end motifs; CNA=copy-number aberration; TMD=targeted-region methylation density; GWMD=genome-wide methylation density; GWFP=genome-wide fragmentation profiles; FLEN=fragment length. Biological preprocessing/window specifications should be read with the methods of the paper; they were not reconstructed from names.
