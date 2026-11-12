# Release scope

This revision keeps all 38 previously published Python files unchanged and adds
the saved numerical evidence, frozen split definitions and result-file lineage.
Notebook uploads are deferred by the authors. The manuscript PDFs are unchanged.
No code license has been selected; this release does not grant an invented license.

## Deposited

- Python code exports and the executed target-only saved-prediction summary.
- Original fp32 baseline/intervention summary JSONs and component tables.
- Original LOLO and row-variance summaries and row-level predictions.
- E1–E8 CSV/JSON results, including held-out predictions, component summaries,
  quality scores, support/count audits, head results and model comparisons.
- Language-ID classifier predictions in six E5B/E6 tables, with predicted labels,
  confidence and off-target flags, indexed in `manifests/CLASSIFIER_PREDICTIONS.json`.
- Frozen splits and exact 309/67/68 E6/E7 matched-support definitions.
- Original completion manifests plus SHA-256 hashes of both deposited and
  decompressed bytes, and a map from revised paper tables to result files.

## Not established by this deposit

Raw attention tensors, model weights and sentence text are not deposited.
Some completion manifests refer to these omitted upstream artifacts; retained
hashes do not imply they are present in GitHub. The exact manually arranged
manuscript figures are not claimed as fully regenerated artwork.

The supplied paper's statement about notebook files remains pending while
notebook uploads are deferred. No manuscript wording or PDF hyperlink was edited.

## Verification

Check `SHA256SUMS.txt` for all repository file bytes. For each result, use
`manifests/RESULT_PROVENANCE.json`; decompress gzip files before comparing the
recorded `uncompressed_sha256`. Archive members were compared with the completed
run ZIPs. Existing scientific source files were compared byte for byte with the
prior published commit. No new model inference or statistical fit was run.

## Classifier documentation clarification

The existing E5B/E6 language-ID outputs provide predicted language, confidence and
off-target flags. These files have been verified and indexed; no predictions were
newly generated. The paper's broad phrase is documented here as language-ID
classifier outputs, without inventing an additional classifier experiment.
