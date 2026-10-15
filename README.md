# Cross-attention multilingual analysis

Original collection and analysis code accompanying **Language Baselines in
Cross-Attention Allocation for Multilingual Indic Neural Machine Translation:
Target-Dominated Additive Structure and Sentence-Level Variation**.

This repository contains **28 Python source exports** of the original notebook
code cells. It preserves the calculations, configuration values, assertions,
paths and cell order. The only executable syntax conversion is six `%pip` setup
lines rendered as equivalent `get_ipython().run_line_magic('pip', ...)` calls,
with an IPython import. It contains no portability adapters or newly written
analysis routines. `SOURCE_PROVENANCE.json` records the original notebook and
code-cell hashes and identifies each syntax conversion.

## File structure

```text
scripts/
  collectors/
    historical/   # original fp16 IT2 pilot, IT2 holdout and NLLB confirmation
    fp32/         # both-model smoke, IT2 full run and NLLB full run
  analysis/
    primary/      # additive/LOLO analyses and fp32 row-level variance
    historical/   # conservative fields, sink tests, head/SVD and fertility
docs/
  REPRODUCIBILITY.md
SOURCE_PROVENANCE.json
SHA256SUMS.txt
```

## Data versions

| Analysis | Collection used |
|---|---|
| Historical IT2 pilot | fp16, 250 meanings |
| Historical IT2 holdout | fp16, 750 meanings |
| Historical NLLB confirmation | fp16, the same 750 meanings |
| fp32 smoke | 120 meanings drawn only from the 450 baseline block; both models |
| Full fp32 IT2 and NLLB | 450 baseline meanings and 300 disjoint intervention meanings per model |
| Pilot LOLO/additive analysis | Original fp16 250 |
| Holdout and NLLB LOLO/additive analyses | New fp32 450 baselines |

Other historical analysis families retain the original fp16 250 / 750 / 750
inputs. The original NLLB free-generation exclusion and reciprocal matching
rules are preserved; missing attention values are not filled with zero.

## Running the code

Use the original **Google Colab/IPython environment** with Drive mounted and the
saved input directories available. These are source exports, not a new portable
command-line pipeline. Model collection needs a suitable CUDA GPU, Hugging Face
dataset/model access and the original pinned dependencies. FLORES+ access and
authentication remain the user's responsibility; no tokens are included.

After cloning the repository into `/content`, run a script from a Colab cell:

```python
%run /content/cross_attention_multilingual_analysis/scripts/collectors/fp32/pilot_fp32_tag_intervention_smoke.py
```

Review the configuration before execution. The original smoke source retains
the saved `TOKEN_POSITIONS_VERIFIED=True`: **set it to False for a fresh run**,
review both printed token/mask audits, then record approval. The saved value
documents the previous run's human review and must not approve a new run.

Run smoke before the full fp32 collectors. Existing saved outputs can be used
directly by CPU analyses. Run head/layer SVD before head-field followup so the
validated head caches are available. See [reproduction details and script map](docs/REPRODUCIBILITY.md).

## Release scope

Notebook markdown, saved notebook outputs, corpus text, model weights and large
attention tensors are not included. The [reproduction guide](docs/REPRODUCIBILITY.md)
links the original saved input/result folders and explains requirements and
coverage. Generated narrative strings inside original code retain their
historical wording; this README describes the present release.

Preparation checked all script syntax, exact preservation of every original
non-magic code-cell line, script/cell hashes, ZIP integrity and absence of literal
credential tokens. GPU inference was not rerun to prepare these exports.
