# Saved numerical evidence

These files are completed-run artifacts copied without statistical changes.
They include original fp32 baseline/intervention summaries, LOLO outputs,
original row-variance predictions, and E1–E8 numerical tables and prediction
files. Worker source is retained in the Python exports. Redundant incremental
CSV chunks are omitted when their consolidated tables are present.

Files above 256,000 bytes are losslessly stored as `.csv.gz`, `.json.gz` or
`.txt.gz`. Read CSVs directly with `pandas.read_csv(path)` or decompress with
`gzip.open`. The decoded bytes exactly match the original artifact. No numbers
are rounded, refitted or recollected for this deposit.

- [Result provenance](../manifests/RESULT_PROVENANCE.json) records archive,
  original member name, physical deposited path, original and compressed hashes,
  byte lengths, CSV columns and row counts.
- [Paper table sources](../manifests/PAPER_TABLE_SOURCES.json) maps revised
  Tables 5–17 to deposited evidence and hashes.
- [Frozen splits](../manifests/SPLITS.json) makes the original 450/300 split,
  strict-generalization splits and E6/E7 matched support directly discoverable.
- [Classifier predictions](../manifests/CLASSIFIER_PREDICTIONS.json) lists six
  E5B/E6 tables containing language-ID labels, confidence and off-target flags.
- [Deferred items](../manifests/UNRESOLVED_ARTIFACTS.json) records remaining
  author decisions and the resolved classifier-file clarification.

Original completion JSONs may also hash files not deposited here, such as
raw per-head blocks or private generation text. Such hashes are retained as
provenance, not as claims that those files are present. Raw attention tensors,
model weights and corpus/reference/generated sentence strings are excluded.

E1 includes only IT2/NLLB; MADLAD generalization comes from E3. E6/E7 use 444
complete meanings with 309 training, 67 validation and 68 test meanings. E2
uses historical fp16 attention, and E8 the historical fp32 crossed intervention.
The E7 target-only summary is the previously executed secondary CPU summary,
not an extra model fit. E8's joint family has 84 interaction cells; 39 of 42
exclude zero in each model, giving 78 of 84 overall.
