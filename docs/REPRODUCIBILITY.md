# Reproduction details



This repository is a Python-only source and result release. Original notebooks are not included. The Python exports preserve the original source logic and runtime assumptions; they are not a new portable command-line pipeline.



Saved numerical results and split manifests are deposited in [results](../results/README.md), [SPLITS.json](../manifests/SPLITS.json), [PAPER_TABLE_SOURCES.json](../manifests/PAPER_TABLE_SOURCES.json), [RESULT_PROVENANCE.json](../manifests/RESULT_PROVENANCE.json), and [REPRODUCIBILITY_INDEX.json](../manifests/REPRODUCIBILITY_INDEX.json).



## CPU-only verification



From the repository root, run:



```bash

python3 scripts/validate_release.py

```



This verifies required files, split sizes, the 444-meaning E6/E7 support, provenance references, and SHA-256 entries. It does not rerun model inference, COMET scoring, tensor analyses, or statistical fits.



## Runtime requirements



- Python 3 with the dependencies retained by the exported setup cells.
- 
- Google Colab/IPython layout for the original exports.
- 
- Drive mounted at the paths retained in the source exports, unless paths are adapted locally.
- 
- CUDA-capable GPU for model collectors and COMET scoring.
- 
- CPU is sufficient for the released analysis scripts when their saved collector inputs are available.
- 
- Pinned Hugging Face model/checkpoint and FLORES+ access described in the source locks and validation guide.
- 
- No access tokens should be committed to the repository.
- 


These exports do not introduce a tested environment lock or portable installation command. Inspect each script's setup cells before execution.



## Collection and execution order



| Stage | Scripts | Inputs / purpose |

|---|---|---|

| Historical collection | `scripts/collectors/historical/indictrans2_2_pilot.py`, `indictrans_2_holdout.py`, `nllb_200_confirmatory.py` | Original fp16 250 / 750 / 750 collectors |

| fp32 validation | `scripts/collectors/fp32/pilot_fp32_tag_intervention_smoke.py` | Both models, 120 baseline-block meanings; needs old holdout metrics for comparison |

| fp32 IT2 | `scripts/collectors/fp32/holdout_fp32_tag_intervention.py` | 40 × 450 baseline TF rows; 168 × 300 crossed TF rows; 28 × 150 output-only generations |

| fp32 NLLB | `scripts/collectors/fp32/nllb_fp32_tag_intervention_confirm.py` | Same frozen meanings and design |

| Additive and LOLO | `scripts/analysis/primary/{pilot,holdout,nllb}_lolo_additive_figure_cpu.py` | Four additive models; held-dyad prediction; language-deletion refits |

| Row variance | `scripts/analysis/primary/03_fp32_row_level_variance_cpu.py` | Both complete fp32 450 × 40 baseline panels |

| Historical field | `scripts/analysis/historical/*conservative_field_final_v3.py` | Field fit, held-out checks, frozen length balancing, and random-edge null |

| Historical followup | `scripts/analysis/historical/*conservative_field_followup_cpu.py` | Historical potential comparisons, covariates, residual/noise tests, additive identity, and NLLB unknown-token audit |

| Historical sink | `scripts/analysis/historical/*additive_sink_cpu.py` | Additive comparison, exact sink accounting, and conditional lexical-share tests |

| Historical heads | `scripts/analysis/historical/*head_layer_svd_cpu.py`, then `*head_field_followup_cpu.py` | Per-head extraction/caches, SVD/null tests, rank-k held-dyad tests, split halves, and localization |

| Historical fertility | `scripts/analysis/historical/*target_fertility_cpu.py` | Target fertility and meaning-clustered within-language regressions |

| E1–E8 validation | `scripts/analysis/validation/` and `scripts/collectors/fp32/` | See [VALIDATION_EXPERIMENTS.md](VALIDATION_EXPERIMENTS.md) for archives, run order, and required saved inputs |



## Supports and denominators



The original fp32 baseline/intervention design uses 450 baseline and 300 disjoint intervention meanings. The strict-generalization split for E1/E3/E4/E5 is 315 train / 67 validation / 68 test.



E6/E7 use 444 common complete meanings from the TF/FG matched panel and retain 309 train / 67 validation / 68 test. Their primary support is 17,760 attention rows per system and regime, with 68 test meanings. Do not silently exchange these denominators with the original 450-meaning anal






