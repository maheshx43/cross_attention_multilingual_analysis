# Reproduction details

Saved numerical results and split manifests are also deposited in this repository:
[results](../results/README.md), [split index](../manifests/SPLITS.json),
[paper-table sources](../manifests/PAPER_TABLE_SOURCES.json) and
[release scope](RELEASE_SCOPE.md).


For the added E1–E8 experiments and the revised-paper table map, see
[VALIDATION_EXPERIMENTS.md](VALIDATION_EXPERIMENTS.md) and
[REVISED_MANUSCRIPT_MAP.md](REVISED_MANUSCRIPT_MAP.md). This guide retains
the original collection and supporting-analysis provenance.

## Collection and execution order

| Stage | Scripts | Inputs / purpose |
|---|---|---|
| Historical collection | `scripts/collectors/historical/indictrans2_2_pilot.py`, `indictrans_2_holdout.py`, `nllb_200_confirmatory.py` | Original fp16 250 / 750 / 750 collectors |
| fp32 validation | `scripts/collectors/fp32/pilot_fp32_tag_intervention_smoke.py` | Both models, 120 baseline-block meanings; needs old holdout metrics for comparison |
| fp32 IT2 | `scripts/collectors/fp32/holdout_fp32_tag_intervention.py` | 40 × 450 baseline TF rows; 168 × 300 crossed TF rows; 28 × 150 output-only generations |
| fp32 NLLB | `scripts/collectors/fp32/nllb_fp32_tag_intervention_confirm.py` | Same frozen meanings and design |
| Additive and LOLO | `scripts/analysis/primary/{pilot,holdout,nllb}_lolo_additive_figure_cpu.py` | Four additive models; held-dyad prediction; language-deletion refits |
| Row variance | `scripts/analysis/primary/03_fp32_row_level_variance_cpu.py` | Both complete fp32 450 × 40 baseline panels |
| Historical field | `scripts/analysis/historical/*conservative_field_final_v3.py` | Field fit, held-out checks, frozen length balancing and random-edge null |
| Historical followup | `scripts/analysis/historical/*conservative_field_followup_cpu.py` | Historical potential comparisons, covariates, residual/noise tests, additive identity and NLLB unknown-token audit |
| Historical sink | `scripts/analysis/historical/*additive_sink_cpu.py` | Additive comparison, exact sink accounting and 100-meaning conditional lexical-share tests |
| Historical heads | `scripts/analysis/historical/*head_layer_svd_cpu.py`, then `*head_field_followup_cpu.py` | Per-head extraction/caches, SVD/null tests, rank-k held-dyad tests, split halves and localization |
| Historical fertility | `scripts/analysis/historical/*target_fertility_cpu.py` | Target fertility and meaning-clustered within-language regressions |

Paths are shown in the original source. CPU analysis scripts require saved
collector metrics; tensor-based tests also need eligible LHTS tensors or their
validated derived head caches. Summary JSON alone cannot reconstruct per-meaning
bootstraps or tensor-level tests. Some cross-collection CPU analyses load all
three historical collections even when the script name selects one output.

Historical collectors resolve revisions into `manifests/source_lock.json`.
Supply the original source lock in the collector root before an independent
reproduction; without it their first-run code resolves remote `main` revisions.
The fp32 collector source pins model/corpus/normalization revisions and writes
the frozen split, run locks, seeds and output checkpoints. The original setup
cells specify package installs; no new environment lock has been invented.

## Deposited results and rerun inputs

- [Original baseline and intervention summaries](../results/original/fp32_tag_intervention)
- [Original LOLO/additive results](../results/original/lolo)
- [Original row-variance results and predictions](../results/original/row_level_variance)
- [E1–E8 validation results](../results/validation)
- [Split definitions](../manifests/SPLITS.json) and [result-file provenance](../manifests/RESULT_PROVENANCE.json)
- [Language-ID classifier predictions](../manifests/CLASSIFIER_PREDICTIONS.json)

The deposited files can be verified without an external shared-folder account.
Full inference and tensor analyses still require the appropriate model/corpus
access and any raw head/tensor inputs not deposited here. Regenerate these with
the collectors or obtain them from the authors. Original scripts retain their
`/content/drive/MyDrive/` runtime paths; recreate that layout or edit their path
configuration in your own runtime. These path defaults are not public data links.
Decompress numerical `.gz` files before placing them at a script's expected
plain CSV/JSON path. Summary JSON alone does not replace raw tensor inputs.

## NLLB missing-row policy

The historical free-generation collector has 57 capped direction–meaning rows.
Their attention statistics are excluded while their output diagnostics remain.
The resulting eligible attention support is 29,943 / 30,000 direction–meaning
rows and reciprocal complete-case support is 14,944 / 15,000 dyad–meaning pairs.
Exclusion and matching are enforced in source; do not impute missing values.
These exclusions do not remove rows from the complete fp32 450-meaning TF panels.

## Manuscript coverage and limitations

The primary scripts implement the direction-level additive/held-dyad and
language-deletion analyses and row-level variance analysis. The fp32 collectors
also compute the baseline field/noise analyses, component decomposition,
crossed tag/text contrasts and generation manipulation checks. The historical
scripts implement the supporting conservative-field, length/null, normalized
lexical, head/SVD and fertility analyses.

This faithful source release does not add replacement code for manuscript
artwork or statistics not explicitly computed in these original notebooks.
In particular, the separate fp32 cross-model point-correlation calculation,
the exact nine language-family contrasts, and the original layout of manuscript
Figures 1–4 are not supplied as new routines here. Original LOLO β/R² plots are
supporting plots, not the manuscript's Figure 2. The row-variance script produces
its variance-partition plot. Full manuscript-figure recreation is not promised.

The deposited source exports and saved numerical artifacts have separate
provenance manifests. The current release scope, including any deferred uploads
and omitted raw inputs, is documented in [RELEASE_SCOPE.md](RELEASE_SCOPE.md).
