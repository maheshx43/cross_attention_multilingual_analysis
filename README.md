# Cross-attention multilingual analysis

Collection and analysis code accompanying **Language-Dependent Cross-Attention in Multilingual Indic Neural Machine Translation**.

This repository is a **Python-only source and result release**. It contains faithful Python exports of notebook code cells, saved numerical results, frozen split definitions, provenance manifests, and validation utilities. The original notebooks are not included in this release. No license is asserted by this repository; users should contact the authors regarding reuse and redistribution.

The calculations, configurations, assertions, paths, and cell order are retained. IPython `%pip` and `%%writefile` syntax is represented by equivalent Python-compatible export logic with explicit staging-path interpolation; worker-file contents are preserved where applicable. `SOURCE_PROVENANCE.json` records source-notebook names, cell ranges, script hashes, and syntax conversions. No scientific routine was rewritten and no GPU experiment was rerun to prepare this release.

## Repository structure

```text
scripts/
  collectors/
    historical/   # original fp16 IT2 pilot, IT2 holdout and NLLB confirmation
    fp32/         # original tag/text collectors; E3, E4, E5B and E6
  analysis/
    primary/      # additive/LOLO analyses and original fp32 row variance
    historical/   # field, sink, head/SVD and fertility analyses
    validation/   # E1, E2, E5A, E7, E8 and saved-prediction summary
manifests/        # frozen splits, result hashes, table/figure provenance
results/          # saved numerical tables, JSON summaries and predictions
docs/             # reproduction, validation, release scope and paper map
SOURCE_PROVENANCE.json
SHA256SUMS.txt
CITATION.cff
RELEASE_NOTES.md
```

## Reproduction scope

Use the original Google Colab/IPython environment for collection and GPU analyses. These are faithful source exports, not a new portable command-line pipeline. The setup cells retain their original dependency pins. Collectors and COMET scoring require suitable CUDA hardware, the pinned Hugging Face model access, FLORES+ access, and the external input directories described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Do not commit access tokens.

For CPU-only verification of deposited results and release metadata, run:

```bash
python3 scripts/validate_release.py
```

The validator checks required files, split sizes, E6/E7 support, paper-table provenance paths, result-manifest paths, and SHA-256 entries. It does not rerun model inference, COMET scoring, tensor analyses, or statistical fits.

For a source export in Colab, run an experiment only in its documented runtime and inspect its configuration first. For example:

```python
%run /content/cross_attention_multilingual_analysis/scripts/analysis/validation/E1_fp32_meaning_dyad_generalization_cpu.py
```

The experiment order, inputs, runtime requirements, archives, and limitations are in the [validation guide](docs/VALIDATION_EXPERIMENTS.md). E4 v3 is the final deterministic collector; E5A precedes E5B; E6 precedes E7; E2 and E8 use their declared historical saved inputs.

## Data, splits, and supports

The historical fp16 analyses use 250 / 750 / 750 meanings. The original fp32 collectors use 450 baseline and 300 disjoint intervention meanings. The original strict-generalization split is **315 training / 67 validation / 68 test** within the 450-meaning baseline block.

E6/E7 use the **444 common complete meanings** from the matched TF/FG panel and retain the partition as **309 training / 67 validation / 68 test**. The primary E6/E7 attention support is 17,760 rows per system and regime, with 68 test meanings. The support definitions are frozen in [manifests/splits/E6_E7_309_67_68_support.json](manifests/splits/E6_E7_309_67_68_support.json) and [results/validation/E6_fp32_regime_comparison_v1/analysis/E6_primary_support.json](results/validation/E6_fp32_regime_comparison_v1/analysis/E6_primary_support.json).

NLLB historical exclusions are preserved as coded; missing attention is never zero-filled. The NLLB capped-row policy and E6 undefined-attention policy are documented in the reproduction and validation guides.

## Deposited evidence and provenance

Saved CSV/JSON files are the numerical evidence. Use [results/README.md](results/README.md), [manifests/SPLITS.json](manifests/SPLITS.json), [manifests/PAPER_TABLE_SOURCES.json](manifests/PAPER_TABLE_SOURCES.json), [manifests/RESULT_PROVENANCE.json](manifests/RESULT_PROVENANCE.json), and [manifests/REPRODUCIBILITY_INDEX.json](manifests/REPRODUCIBILITY_INDEX.json) to locate the evidence for each manuscript table and figure. Larger files use lossless gzip.

The deposited evidence can be checked without Drive access. Full recollection may still require the pinned models, gated corpus access, raw attention tensors, per-head blocks, external Drive inputs, or the original directory layout. The release explicitly separates directly verifiable saved evidence from analyses that require omitted upstream artifacts.

The exact manually arranged manuscript artwork is not claimed as fully regenerated. Dedicated replacement routines are not supplied for the separate fp32 cross-model point-correlation calculation or the exact nine language-family contrasts; their availability status is recorded in the provenance index and release notes.

## Revised-paper coverage

The [manuscript map](docs/REVISED_MANUSCRIPT_MAP.md) connects revised tables and figures to source exports and saved numerical evidence. E2 reports no reliable incremental COMET prediction benefit in the two tested systems. Corrected E5B selected-versus-random evidence is positive only for MADLAD-400. E7 finds no reliable gain from the tested flexible model forms. E8 reports the crossed tag/text interaction with simultaneous adjustment over all 84 cells.

Language-identification outputs are deposited in the E5B/E6 tables and indexed in [manifests/CLASSIFIER_PREDICTIONS.json](manifests/CLASSIFIER_PREDICTIONS.json). These are existing language-ID predictions, not a new classification experiment.

## Verification and release identity

Run `python3 scripts/validate_release.py` before citing or archiving a checkout. The exact release commit should be recorded with any submission. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the Python-only scope, omitted artifacts, validation limits, and known non-rerunnable analyses.
