# Cross-attention multilingual analysis

Collection and analysis code accompanying **Language-Dependent Cross-Attention
in Multilingual Indic Neural Machine Translation**.

This repository contains **37 Python exports of notebook code cells** (the
original 28 and nine completed E1–E8 validation notebooks), plus **one executed
CPU summary of saved predictions** used for the target-only comparison in
revised Table 8.

The calculations, configurations, assertions, paths and cell order are retained.
IPython `%pip` and `%%writefile` syntax is represented by equivalent IPython
magic calls, with explicit staging-path interpolation for script execution;
worker-file contents are preserved exactly. The original 28
Python scripts are unchanged. `SOURCE_PROVENANCE.json` records notebook, cell
and script hashes and each syntax conversion. No scientific routine was
rewritten and no GPU experiment was rerun to prepare this release.

## File structure

```text
scripts/
  collectors/
    historical/   # original fp16 IT2 pilot, IT2 holdout and NLLB confirmation
    fp32/         # original tag/text collectors; E3, final E4, E5B and E6
  analysis/
    primary/      # additive/LOLO analyses and original fp32 row variance
    historical/   # original field, sink, head/SVD and fertility analyses
    validation/   # E1, E2, E5A, E7, E8 and saved-prediction summary
docs/
  REPRODUCIBILITY.md
  VALIDATION_EXPERIMENTS.md
  REVISED_MANUSCRIPT_MAP.md
results/          # saved numerical tables, JSON summaries and predictions
manifests/        # frozen splits, result hashes and paper-table source map
SOURCE_PROVENANCE.json
SHA256SUMS.txt
```

## Running the code

Use the original **Google Colab/IPython environment**, with Drive mounted and
the saved input directories available. These are faithful source exports,
not a new portable command-line pipeline. The setup cells retain their original
dependency pins. Collectors and COMET scoring need suitable CUDA hardware and
the required Hugging Face model/dataset access. Do not commit access tokens.

After cloning the repository into `/content`, run an export from a Colab cell:

```python
%run /content/cross_attention_multilingual_analysis/scripts/analysis/validation/E1_fp32_meaning_dyad_generalization_cpu.py
```

Run each experiment in its own appropriate runtime and inspect its configuration
before execution. E4 v3 is the final deterministic collector; failed earlier E4
replay notebooks are not released. E5A precedes E5B; E6 precedes E7. E2 and E8
use their declared historical saved inputs. The exact dependencies, runtime
requirements, saved archive names and run order are in the
[validation guide](docs/VALIDATION_EXPERIMENTS.md).

For a fresh run of the original fp32 smoke source, set its saved
`TOKEN_POSITIONS_VERIFIED=True` to `False`, inspect both printed token/mask
audits and only then record approval. That saved flag documents the original
run and does not approve a new run.

## Data and results

The original fp16 analyses use 250 / 750 / 750 meanings. The original fp32
collectors use 450 baseline and 300 disjoint intervention meanings; the new
strict generalization split is 315 / 67 / 68 within the 450 baseline block.
E6/E7 use 444 complete matched meanings with retained 309 / 67 / 68 partitions.
NLLB exclusions are preserved as coded; missing attention is never zero-filled.

Saved split definitions, numerical summaries and prediction files are now
deposited in [results/](results/README.md) and [manifests/](manifests/SPLITS.json).
[Paper-table provenance](manifests/PAPER_TABLE_SOURCES.json) connects revised
Tables 5–17 to the deposited files and hashes. Larger files use lossless gzip.

The original source/result archives and additional upstream inputs remain on Drive:

- [E1–E8 completed result archives](https://drive.google.com/drive/folders/1oAxnr1y1DwMfzvNGEqFijLGiymKFm8OI)
- [Original fp32 baseline and crossed intervention outputs](https://drive.google.com/drive/folders/1IzgV6eRs7nqCU_Ffr9Baoc5aMARrhSHf)
- [Original additive/LOLO outputs](https://drive.google.com/drive/folders/1aLYAwzlTwI0qGwTX1T-UYE_YShLpqYy1)

The deposited CSV/JSON files are the numerical evidence; notebook displays may
truncate tables. GitHub verification of these deposited values does not require
Drive access. Full recollection still needs original upstream inputs, model and
corpus access, and the directory structure documented by the scripts. Raw
attention tensors, model weights and corpus strings are excluded. See the
[reproduction guide](docs/REPRODUCIBILITY.md) for other inputs.

## Revised-paper coverage

The [manuscript map](docs/REVISED_MANUSCRIPT_MAP.md) connects revised sections
and tables to the original code and E1–E8 outputs. Results include negative and
model-dependent findings: E2 establishes no reliable incremental COMET
prediction benefit; corrected E5B selected-versus-random evidence is positive
only for MADLAD; E7 finds no reliable gain from the tested flexible models.
E8 has 39/42 nonzero interaction intervals per model (78/84 overall), adjusted
jointly across all 84 cells.

The [release scope](docs/RELEASE_SCOPE.md) records the deposited material,
verification procedure and items deferred by the authors. Classifier prediction
files have not been located or verified; no such file is claimed here.
