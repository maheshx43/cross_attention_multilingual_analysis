# Reproduction details

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

## Saved input and result directories

- [Historical collector backups](https://drive.google.com/drive/folders/1VIuI5ensLAlLiMYO3nhCDxXYYm8cho71)
- [fp32 baseline, intervention and row-variance outputs](https://drive.google.com/drive/folders/1IzgV6eRs7nqCU_Ffr9Baoc5aMARrhSHf)
- [LOLO/additive outputs](https://drive.google.com/drive/folders/1aLYAwzlTwI0qGwTX1T-UYE_YShLpqYy1)
- [Historical head/SVD outputs and caches](https://drive.google.com/drive/folders/1tdYS7-ks_u1IBWDt87yIEXn_fgmmt9Hg)

These external folders are separate from this source release. Their availability
depends on Drive permissions. Reproduce the directory structure in the source
under `/content/drive/MyDrive/`, or explicitly edit the path configuration for
your own runtime. Some external files contain reference/generated text; obtain
the appropriate dataset access before using them.

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

For a manuscript availability statement, describe this repository as **Python
source exports of the collection and analysis notebooks**, with saved data and
results available separately subject to access requirements. Do not state that
this repository contains the notebooks, all raw data, or every figure source.
