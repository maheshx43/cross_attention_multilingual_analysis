# Release scope



This is a **Python-only** source and result release. It keeps the existing scientific source exports and adds saved numerical evidence, frozen split definitions, result-file lineage, citation metadata, and CPU-only release validation. Original `.ipynb` files are not included. No software license is asserted by this repository.



## Deposited



- Python source exports and the executed target-only saved-prediction summary.
- 
- Original fp32 baseline/intervention summary JSONs and component tables.
- 
- Original LOLO and row-variance summaries and row-level predictions.
- 
- E1–E8 CSV/JSON results, including held-out predictions, component summaries, quality scores, support/count audits, head results, and model comparisons.
- 
- Language-ID classifier predictions in six E5B/E6 tables, with predicted labels, confidence, and off-target flags, indexed in `manifests/CLASSIFIER_PREDICTIONS.json`.
- 
- Frozen splits and exact 309/67/68 E6/E7 matched-support definitions.
- 
- Completion manifests, SHA-256 hashes, paper-table provenance, and the machine-readable `manifests/REPRODUCIBILITY_INDEX.json`.
- 
- `CITATION.cff`, `RELEASE_NOTES.md`, and `scripts/validate_release.py`.
- 


## Not established by this deposit



Raw attention tensors, model weights, sentence text, external Drive inputs, and some per-head blocks are not deposited. Some completion manifests refer to omitted upstream artifacts; retained hashes do not imply that those artifacts are present in GitHub. Full recollection therefore requires the pinned upstream models/corpus and the documented source directory layout.



The exact manually arranged manuscript figures are not claimed as fully regenerated artwork. Dedicated replacement routines are not supplied for the separate fp32 cross-model point-correlation calculation or the exact nine language-family contrasts. These boundaries are recorded in `manifests/REPRODUCIBILITY_INDEX.json` and `docs/REPRODUCIBILITY.md`.



## Verification



Run `python3 scripts/validate_release.py` to verify required files, split sizes, E6/E7 support counts, provenance paths, and SHA-256 entries. This validation does not run model inference, COMET scoring, tensor analyses, or statistical fits.



## Reuse and redistribution



No software license has been selected. The repository does not grant an implied license; contact the authors regarding reuse and redistribution.



## Submission identity



For an IEEE submission, record the exact commit hash or release tag used for the submission. Do not describe notebooks as deposited unless they are added in a later release.









