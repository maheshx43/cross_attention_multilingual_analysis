# Completed validation experiments

Saved numerical results and split manifests are also deposited in this repository:
[results](../results/README.md), [split index](../manifests/SPLITS.json),
[paper-table sources](../manifests/PAPER_TABLE_SOURCES.json) and
[release scope](RELEASE_SCOPE.md).


These exports come from the completed Colab notebooks, rather than the earlier
unexecuted development copies. E5 has two notebooks, making nine E1–E8 exports.
Notebook markdown and outputs are intentionally excluded. Code-generated
narrative strings remain unchanged as part of the executed source; interpret
completion using the later experiment and saved completion manifests.

## Scripts and run order

| Stage | Python export | Runtime and saved inputs | Completed result archive |
|---|---|---|---|
| E1 | [Meaning/dyad generalization](../scripts/analysis/validation/E1_fp32_meaning_dyad_generalization_cpu.py) | CPU; original fp32 IT2/NLLB baseline450 | `E1_generalization_cpu_v1_results.zip` |
| E2 | [COMET quality validation](../scripts/analysis/validation/E2_COMET_quality_validation_L4.py) | L4 for COMET and likelihood; CPU regression; historical fp16 generated-output attention | `E2_quality_comet_v1_results.zip` |
| E3 | [MADLAD validation](../scripts/collectors/fp32/E3_MADLAD_fp32_validation_L4.py) | L4; third-model fp32 TF collection; E1 and original fp32 inputs | `E3_madlad_fp32_v1_results.zip` |
| E4 | [Deterministic sensitivity v3](../scripts/collectors/fp32/E4_deterministic_v3.py) | L4 collection; CPU analysis; E3 and baseline source identities | `E4_mask_sensitivity_fp32_v3_results.zip` |
| E5A | [Head analysis](../scripts/analysis/validation/E5_fp32_head_analysis_cpu.py) | CPU; E4 saved per-head blocks, not a new model load | `E5_fp32_head_analysis_v1_results.zip` |
| E5B | [Head ablation and quality](../scripts/collectors/fp32/E5_head_ablation_quality_L4.py) | L4; E4 inputs and E5A frozen selected/random head sets | `E5_head_ablation_fp32_v1_results.zip` |
| E6 | [TF/free-generation comparison](../scripts/collectors/fp32/E6_fp32_TF_free_generation_L4.py) | L4; E4 TF baseline, original meanings, E5B robustness readback | `E6_fp32_regime_comparison_v1_results.zip` |
| E7 | [Additive/interaction/nonlinear comparison](../scripts/analysis/validation/E7_additive_interaction_nonlinear_cpu.py) | CPU; E6 matched panel and strict held-out folds | `E7_fp32_model_comparison_v1_results.zip` |
| E8 | [Crossed tag/text interaction](../scripts/analysis/validation/E8_crossed_tag_text_interaction_cpu.py) | CPU; original IT2/NLLB crossed interventions, original split and frozen beta | `E8_crossed_tag_text_cpu_v1_results.zip` |

The completed numerical artifacts are deposited under
[results/validation](../results/validation). Each experiment directory retains
the original result names and completion manifests; larger files add a `.gz`
suffix. [RESULT_PROVENANCE.json](../manifests/RESULT_PROVENANCE.json) records
the source archive/member and both deposited and decompressed hashes.
Archive names above identify run provenance rather than an external download
requirement. Restore the script's configured directory layout when rerunning.
Raw per-head blocks and gated corpus/model inputs remain separate requirements
for full recollection; they are not replaced by numerical summaries.

## Secondary target-only summary

[complete_e7_target_baseline.py](../scripts/analysis/validation/complete_e7_target_baseline.py)
is the already executed CPU summary used for the target-only column in revised
Table 8. It summarizes saved E6 target-only predictions with the identical E7
joint meaning/dyad draws. It first reproduces the saved M1 point estimates and
intervals exactly; it performs no new fit or inference. It was separate from the
nine Colab runs and is released without path or numerical changes.

The script expects `results_cache/` beside itself, containing extracted
`E4_mask_sensitivity_fp32_v3`, `E6_fp32_regime_comparison_v1` and
`E7_fp32_model_comparison_v1` directories. It writes
`E7_target_only_report_summary.csv` and `E7_target_only_report_validation.json`
beside itself. Its NumPy/pandas environment must be supplied; source export
validation is not a fresh statistical rerun.

## Scope and important distinctions

- E1 contains IT2/NLLB only. MADLAD strict-generalization values, including
  individual-row R2 = 0.3286, come from E3. E6/E7 use the later matched panel;
  do not silently exchange denominators or call their estimates identical.
- E2 uses pinned COMET22-DA (`Unbabel/wmt22-comet-da`, revision `2760a223`).
  All 60,000 translations are scored, while attention predictors retain their
  historical eligibility. Likelihood is a retokenized content score. Optional
  human ratings were not collected; no human-validation claim is supported.
- E4 v3 freezes `PYTHONHASHSEED=0` at process start and saves actual token IDs.
  Raw and conditional lexical outcomes differ. Target-coefficient invariance
  under source-only covariates can follow algebraically from the balanced panel.
- E5A's historical ablation-pending message is superseded by completed E5B.
  The executed E5A source includes its `MPLBACKEND=Agg` fix. Whole-head ablation
  does not isolate EOS or a single-head causal mechanism.
- E6 retains all 54,000 behavioral outputs. Seven undefined attention rows
  affect six meanings, leaving 444 common complete meanings. All-original,
  matched, and globally uncapped supports are reported separately.
- Some E6/E7 direction-level bootstrap distributions are biased. The saved
  uncertainty audit does not establish a validated replacement BCa interval.
  The revised paper reports lower percentiles and describes this limitation;
  do not relabel the saved intervals or recompute convenient replacements.
- E8 uses the original historical fp32 intervention outputs; it is not a fresh
  deterministic E4-style collection. Its interval family is 7 tags x 6 forced
  texts x 2 models = 84 tag/text cells. Under the joint simultaneous adjustment,
  39 of 42 exclude zero in each model, or 78 of 84 overall. The smaller-control
  shift prediction fails for NLLB. Projection ratios are not causal mediation.

## Export and reproduction checks

`SOURCE_PROVENANCE.json` gives source-notebook hashes, code-cell byte ranges,
the hash of each exported block and any IPython-only syntax conversion. The
`%%writefile` payloads are checked byte for byte against the executed notebook;
their `{STAGING}` paths are explicitly interpolated in the script namespace so
`%run` writes to the same configured directory as the notebook.
The standalone saved-prediction script is also copied byte for byte. The
original 28 exports are retained byte for byte. `SHA256SUMS.txt` covers the
published files other than itself. GPU collectors and COMET are not rerun as
part of preparing this source release.

For verification of reported values, use full saved CSV/JSON files and their
completion hashes, not truncated notebook displays. For a full rerun, supply
the upstream Drive files, per-head blocks when needed, pinned model/corpus
access and compatible hardware. The exports retain the original Colab setup
and original paths instead of introducing an untested portability layer.

## Language-identification classifier outputs

[CLASSIFIER_PREDICTIONS.json](../manifests/CLASSIFIER_PREDICTIONS.json) identifies
six deposited E5B/E6 tables with per-output predicted language (`lid_label`),
confidence (`lid_probability`) and requested-language mismatch (`off_target`).
These are existing language-ID classifier predictions, not an additional
classification experiment. E2 predicts continuous quality scores by regression.
