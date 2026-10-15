# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 2
# 0. Fresh root, dependencies, Drive restore, and source-lock location
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = "/content/nllb_40_confirmatory_holdout_750_final_v6"
AUTO_DRIVE_BACKUP = True
DRIVE_SYNC_EVERY_PAIRS = 2
DRIVE_BACKUP_BASE = Path("/content/drive/MyDrive/indic_40_backups")

Path(ROOT_DIR).mkdir(parents=True, exist_ok=True)
if AUTO_DRIVE_BACKUP:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DESTINATION = DRIVE_BACKUP_BASE / Path(ROOT_DIR).name
    local_has_progress = (
        any(Path(ROOT_DIR).glob("teacher_forced/*_stats.csv"))
        or any(Path(ROOT_DIR).glob("free_generation/*_stats.csv"))
    )
    if DRIVE_DESTINATION.is_dir() and not local_has_progress:
        subprocess.run([
            "rsync", "-r", "--checksum", str(DRIVE_DESTINATION) + "/",
            str(Path(ROOT_DIR)) + "/",
        ], check=True)
        print("Restored incremental checkpoint from", DRIVE_DESTINATION)
else:
    DRIVE_DESTINATION = None

Path(ROOT_DIR, "manifests").mkdir(parents=True, exist_ok=True)
SOURCE_LOCK_PATH = Path(ROOT_DIR, "manifests", "source_lock.json")
SOURCE_LOCK = json.loads(SOURCE_LOCK_PATH.read_text()) if SOURCE_LOCK_PATH.exists() else {}

def save_source_lock():
    temporary = SOURCE_LOCK_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(SOURCE_LOCK, indent=2), encoding="utf-8")
    temporary.replace(SOURCE_LOCK_PATH)

subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
    "--upgrade-strategy", "only-if-needed",
    "transformers==4.51.3", "tokenizers==0.21.1",
    "huggingface_hub==0.30.2", "datasets==4.0.0",
    "sacrebleu==2.3.1", "sentencepiece==0.2.2",
], check=True)
print("Pinned NLLB/FLORES dependencies installed.")


# %% original code cell 1; notebook cell 3
# ============================================================
# 1. Imports, deterministic controls, and immutable run configuration
# ============================================================
import gc
import hashlib
import platform
import re
import random
import shutil
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

SEED = 42
PIPELINE_SCHEMA_VERSION = "nllb-cross-attention-1.4.0"
PIPELINE_SOURCE_SHA256 = "bf6718688d1c56a4f17d826d093f543ec8b1b235303f614f829f79698362ae78"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

seed_everything()
STRICT_DETERMINISM = False
if STRICT_DETERMINISM:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_NAME = "facebook/nllb-200-distilled-600M"
# Exact public checkpoint resolved on 2026-09-13; shared by smoke and holdout.
MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
DATASET_NAME = "openlanguagedata/flores_plus"
DATASET_SPLIT = "devtest"
# Exact dataset commit used by the completed IndicTrans2 holdout.
DATASET_REVISION = "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"

CORPUS_ROW_START = 250
CORPUS_ROW_STOP_EXCLUSIVE = 1000
N_SENTENCES = 750
FREE_N_SENTENCES = N_SENTENCES
FREE_SENTENCE_IDS = tuple(range(FREE_N_SENTENCES))
SELECTION_ROLE = 'cross_model_confirmatory_holdout'
EXPECTED_INDIC_SAMPLE_IDENTITY_SHA256 = '64847ea1866cf6e728315df46e3743f8f69f94614cfde895f53f748ff9d7953c'
assert CORPUS_ROW_STOP_EXCLUSIVE - CORPUS_ROW_START == N_SENTENCES

MAX_LENGTH = 256
FREE_MAX_LENGTH = 256
RUN_MODE = "both"                 # both | teacher_forced_only | free_generation_only
RUN_TEACHER_FORCED = RUN_MODE in {"both", "teacher_forced_only"}
RUN_FREE_GENERATION = RUN_MODE in {"both", "free_generation_only"}

BATCH_SIZE_TF = 32
BATCH_SIZE_FREE = 1
PRINT_EVERY_TF_BATCHES = 4
PRINT_EVERY_FREE_ITEMS = 10
RESUME = True
SAVE_FULL_TENSORS_FREE = True
GREEDY_NUM_BEAMS = 1
DO_SAMPLE = False

# Matched exploratory audit, preserving the IndicTrans2 beam-3 condition.
RUN_BEAM_AUDIT = True
BEAM_NUM_BEAMS = 3
BEAM_MAX_LENGTH = 256
BEAM_AUDIT_N = min(50, N_SENTENCES)
BEAM_BATCH_SIZE = 8

DISK_SAFETY_FACTOR = 1.25
DISK_FIXED_OVERHEAD_GB = 10
MIN_FREE_VRAM_GB = 8
REQUIRED_GPU_NAME = "NVIDIA L4"
REQUIRED_TORCH_VERSION = "2.11.0+cu128"
REQUIRED_TORCH_CUDA = "12.8"
INDICTRANS_HOLDOUT_DRIVE_ROOT = str(
    Path("/content/drive/MyDrive/indic_40_backups")
    / "indic_40_balanced_holdout_750_rows250_999_locked_v4"
)

for sub in ("teacher_forced", "free_generation", "free_generation_capped", "logs", "manifests"):
    os.makedirs(os.path.join(ROOT_DIR, sub), exist_ok=True)

CORE_CONFIG = {
    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
    "pipeline_source_sha256": PIPELINE_SOURCE_SHA256,
    "model": MODEL_NAME,
    "model_revision_requested": MODEL_REVISION,
    "dataset": DATASET_NAME,
    "dataset_split": DATASET_SPLIT,
    "dataset_revision_requested": DATASET_REVISION,
    "selection_role": SELECTION_ROLE,
    "corpus_row_start": CORPUS_ROW_START,
    "corpus_row_stop_exclusive": CORPUS_ROW_STOP_EXCLUSIVE,
    "n_sentences": N_SENTENCES,
    "free_n_sentences": FREE_N_SENTENCES,
    "free_sentence_ids": list(FREE_SENTENCE_IDS),
    "max_length": MAX_LENGTH,
    "free_max_length": FREE_MAX_LENGTH,
    "length_semantics": "total decoder sequence length, including decoder start",
    "cap_policy": {
        "translation_quality": "retain capped hypothesis as observed model failure",
        "attention": "exclude capped row from inference",
        "tensor_storage": "free_generation_capped diagnostic directory only",
        "reciprocal_analysis": "complete cases where both directions are eligible",
    },
    "batch_sizes": {"tf": BATCH_SIZE_TF, "free": BATCH_SIZE_FREE},
    "save_full_tensors_free": SAVE_FULL_TENSORS_FREE,
    "tokenizer_legacy_behaviour": False,
    "tokenizer_use_fast": False,
    "target_row_alignment": {
        "teacher_forced": "predicted_label_token",
        "free_generation": "predicted_generated_token",
    },
    "greedy_num_beams": GREEDY_NUM_BEAMS,
    "greedy_num_return_sequences": 1,
    "do_sample": DO_SAMPLE,
    "seed": SEED,
    "strict_determinism": STRICT_DETERMINISM,
    "automatic_drive_backup": AUTO_DRIVE_BACKUP,
    "drive_sync_every_pairs": DRIVE_SYNC_EVERY_PAIRS,
    "drive_backup_base": str(DRIVE_BACKUP_BASE),
    "disk_safety_factor": DISK_SAFETY_FACTOR,
    "disk_fixed_overhead_gib": DISK_FIXED_OVERHEAD_GB,
    "beam_audit": {"enabled": RUN_BEAM_AUDIT, "n": BEAM_AUDIT_N,
                   "num_beams": BEAM_NUM_BEAMS, "max_length": BEAM_MAX_LENGTH},
    "required_runtime": {
        "gpu": REQUIRED_GPU_NAME,
        "torch": REQUIRED_TORCH_VERSION,
        "torch_cuda": REQUIRED_TORCH_CUDA,
    },
    "indictrans_holdout_drive_root_for_common_quality": INDICTRANS_HOLDOUT_DRIVE_ROOT,
    "collector_inferential_tests": "none",
    "planned_external_cross_model_analysis": {
        "status": "pre-specified here; calculated only in a separate shared statistics notebook",
        "primary_endpoint": "mirror_reciprocity_delta_jsd",
        "grid_size": 20,
        "matrix_preparation": (
            "lexical tensor; normalize each target row over source; bilinear "
            "interpolation to 20x20"
        ),
        "mirror_operation": "transpose reverse-direction target-by-source grid",
        "metric": "natural-log Jensen-Shannon divergence after joint normalization",
        "layer_head_aggregation": "compute JSD per layer/head, then arithmetic mean",
        "sentence_aggregation": "arithmetic mean within each reciprocal dyad and regime",
        "outcome": "delta_jsd = free_generation_jsd - teacher_forced_jsd",
        "contrast": "mean_cross_family_delta_jsd - mean_within_family_delta_jsd",
        "directional_prediction": "positive",
        "inferential_units": "20 reciprocal language dyads on 11 language nodes",
        "primary_test": (
            "exact two-sided graph-preserving node-label permutation over all "
            "C(11,4)=330 assignments"
        ),
        "required_robustness": "positive contrast after every leave-one-language-out omission",
        "secondary_replication_endpoint": {
            "metric": "diagonal_excess_10pct",
            "outcome": "free_generation_minus_teacher_forced",
            "contrast": "mean_within_family_minus_mean_cross_family",
            "role": "secondary; multiplicity handling is external",
        },
        "multiplicity_policy": (
            "no correction for the single frozen mirror-JSD primary; all secondary "
            "and supporting tests corrected in the separate shared statistics notebook"
        ),
    },
    "metric_protocol": "FLORES-200 spBLEU; raw-text chrF and chrF++",
    "unknown_token_policy": (
        "retain tokenizer <unk> as an active lexical placeholder; audit source, "
        "reference and generated occurrences; include generated <unk> in quality flags"
    ),
    "repetition_policy": (
        "repeated n-grams only; character-run and generated-subword diagnostics; "
        "short unique hypotheses are not flagged solely by denominator size"
    ),
}

config_path = os.path.join(ROOT_DIR, "manifests", "run_config.json")
if os.path.isfile(config_path):
    with open(config_path, encoding="utf-8") as handle:
        previous_config = json.load(handle)
    assert previous_config == CORE_CONFIG, (
        "ROOT_DIR contains an incompatible NLLB configuration. Use its matching "
        "checkpoint or a fresh ROOT_DIR; never mix protocols."
    )
else:
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(CORE_CONFIG, handle, indent=2, ensure_ascii=False)

print("Device:", DEVICE, "| dtype:", DTYPE)
print("Root:", ROOT_DIR)
print("Selection:", CORPUS_ROW_START, "to", CORPUS_ROW_STOP_EXCLUSIVE - 1)
print("Protocol:", PIPELINE_SCHEMA_VERSION, PIPELINE_SOURCE_SHA256)


# %% original code cell 2; notebook cell 4
# ============================================================
# 2. Non-interactive Hugging Face authentication
# ============================================================
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN") or HF_TOKEN
except Exception:
    pass

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Hugging Face authentication loaded non-interactively.")
else:
    print("HF_TOKEN not set; continuing without a login for public artifacts.")

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
def locked_hf_revision(repo, requested, kind):
    key = f"hf:{kind}:{repo}:{requested}"
    if key not in SOURCE_LOCK:
        info = (api.model_info(repo, revision=requested) if kind == "model"
                else api.dataset_info(repo, revision=requested))
        SOURCE_LOCK[key] = info.sha
        save_source_lock()
    return SOURCE_LOCK[key]

RESOLVED_MODEL_COMMIT = locked_hf_revision(MODEL_NAME, MODEL_REVISION, "model")
RESOLVED_DATASET_COMMIT = locked_hf_revision(DATASET_NAME, DATASET_REVISION, "dataset")


# %% original code cell 3; notebook cell 5
# ============================================================
# 2b. Load NLLB with explicit default (non-legacy) tokenization
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, revision=RESOLVED_MODEL_COMMIT,
    src_lang="eng_Latn", tgt_lang="hin_Deva",
    legacy_behaviour=False, use_fast=False,
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, attn_implementation="eager",
    token=HF_TOKEN, revision=RESOLVED_MODEL_COMMIT,
).to(DEVICE).eval()

def language_id(lang):
    value = int(tokenizer.convert_tokens_to_ids(lang))
    assert value != int(tokenizer.unk_token_id), f"Unsupported NLLB language code: {lang}"
    assert tokenizer.convert_ids_to_tokens(value) == lang
    return value

def configure_languages(src_lang, tgt_lang):
    """Make mutable NLLB tokenizer language state explicit before every call."""
    language_id(src_lang); language_id(tgt_lang)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    tokenizer.set_src_lang_special_tokens(src_lang)

def token_pieces(ids):
    return tokenizer.convert_ids_to_tokens([int(x) for x in ids])

print("model_max_length:", tokenizer.model_max_length)
print("Tokenizer:", type(tokenizer).__name__, "legacy_behaviour=", tokenizer.legacy_behaviour)
print("Model loaded on", DEVICE, "dtype", next(model.parameters()).dtype)
assert tokenizer.legacy_behaviour is False
assert getattr(model.config, "_commit_hash", None) == RESOLVED_MODEL_COMMIT
assert model.config.is_encoder_decoder
print("Resolved model commit:", RESOLVED_MODEL_COMMIT)


# %% original code cell 4; notebook cell 6
# ============================================================
# 3. NLLB language coverage and special-token probes
# ============================================================
TARGET_LANGS = {
    "hin_Deva", "ben_Beng", "mar_Deva", "guj_Gujr", "pan_Guru", "ory_Orya",
    "asm_Beng", "tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym",
}
language_ids = {lang: language_id(lang) for lang in sorted(TARGET_LANGS)}
assert len(set(language_ids.values())) == len(TARGET_LANGS)

_PROBE_TEXT_RAW = "यह एक परीक्षण वाक्य है ।"
_PROBE_TGT_TEXT = "ಇದು ಒಂದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ."
_PROBE_SRC, _PROBE_TGT = "hin_Deva", "kan_Knda"

def tokenizer_state_probe(tag=""):
    configure_languages(_PROBE_SRC, _PROBE_TGT)
    source = tokenizer(_PROBE_TEXT_RAW, add_special_tokens=True,
                       return_attention_mask=True)
    target = tokenizer(text_target=_PROBE_TGT_TEXT, add_special_tokens=True,
                       return_attention_mask=True)
    src_ids = source["input_ids"]
    tgt_ids = target["input_ids"]
    assert src_ids[0] == language_id(_PROBE_SRC) and src_ids[-1] == tokenizer.eos_token_id
    assert tgt_ids[0] == language_id(_PROBE_TGT) and tgt_ids[-1] == tokenizer.eos_token_id
    # text_target must return tokenizer state to source mode.
    again = tokenizer(_PROBE_TEXT_RAW, add_special_tokens=True)["input_ids"]
    assert again == src_ids
    print(f"[probe @ {tag}] source={token_pieces(src_ids[:4])} ... {token_pieces(src_ids[-2:])}")
    print(f"[probe @ {tag}] target={token_pieces(tgt_ids[:4])} ... {token_pieces(tgt_ids[-2:])}")
    return tuple(src_ids), tuple(tgt_ids)

_probe_baseline = tokenizer_state_probe("baseline")
print("All 11 required NLLB language codes are distinct and supported.")


# %% original code cell 5; notebook cell 7
# ============================================================
# 3b. Official NLLB generation smoke test
# ============================================================
configure_languages(_PROBE_SRC, _PROBE_TGT)
inputs = tokenizer(
    [_PROBE_TEXT_RAW], truncation=True, padding="longest",
    return_tensors="pt", return_attention_mask=True,
).to(DEVICE)
seed_everything(SEED)
with torch.inference_mode():
    generated = model.generate(
        **inputs,
        forced_bos_token_id=language_id(_PROBE_TGT),
        use_cache=True, min_length=0, max_length=FREE_MAX_LENGTH,
        num_beams=1, num_return_sequences=1, do_sample=False,
    )
sequence = generated[0].detach().cpu().tolist()
assert sequence[0] == model.config.decoder_start_token_id
assert sequence[1] == language_id(_PROBE_TGT)
translations = tokenizer.batch_decode(
    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
assert translations[0].strip()
assert tokenizer_state_probe("after generation") == _probe_baseline
SMOKE_TRANSLATION_PASSED = True
print("Smoke translation:", translations[0])
print("NLLB forced-target-language generation passed.")


# %% original code cell 6; notebook cell 8
# 4. Load the exact FLORES+ revision
ds = load_dataset(
    DATASET_NAME, split=DATASET_SPLIT, revision=RESOLVED_DATASET_COMMIT
)
DATASET_FINGERPRINT = getattr(ds, "_fingerprint", None)
print(ds)
print("Dataset fingerprint:", DATASET_FINGERPRINT)


# %% original code cell 7; notebook cell 9
# 4b. Exact FLORES row staging
ID_COL, LANG_COL, SCRIPT_COL, TEXT_COL = "id", "iso_639_3", "iso_15924", "text"
df_long = ds.to_pandas()
df_long["lang_code"] = df_long[LANG_COL].astype(str) + "_" + df_long[SCRIPT_COL].astype(str)
df_indic = df_long[df_long["lang_code"].isin(TARGET_LANGS)].copy()
dupes = df_indic[df_indic.duplicated(subset=[ID_COL, "lang_code"], keep=False)]
assert len(dupes) == 0, "Duplicate FLORES id/language rows found."
df_wide_all = (
    df_indic.pivot(index=ID_COL, columns="lang_code", values=TEXT_COL)
    .reset_index().sort_values(ID_COL).reset_index(drop=True)
)
missing = sorted(TARGET_LANGS - set(df_wide_all.columns))
assert not missing, f"Missing required languages: {missing}"
assert CORPUS_ROW_STOP_EXCLUSIVE <= len(df_wide_all)
df_wide = df_wide_all.iloc[CORPUS_ROW_START:CORPUS_ROW_STOP_EXCLUSIVE].reset_index(drop=True)
assert len(df_wide) == N_SENTENCES
print("FLORES shared-meaning table:", df_wide.shape, flush=True)
print("Global corpus rows:", CORPUS_ROW_START, "..", CORPUS_ROW_STOP_EXCLUSIVE - 1)


# %% original code cell 8; notebook cell 10
# ============================================================
# 5. Reciprocal, relation-balanced 40-direction design
# ============================================================
WITHIN_IA = [
    ("asm_Beng", "ben_Beng"), ("ben_Beng", "asm_Beng"),
    ("hin_Deva", "ory_Orya"), ("ory_Orya", "hin_Deva"),
    ("mar_Deva", "guj_Gujr"), ("guj_Gujr", "mar_Deva"),
    ("guj_Gujr", "pan_Guru"), ("pan_Guru", "guj_Gujr"),
    ("ben_Beng", "mar_Deva"), ("mar_Deva", "ben_Beng"),
]
WITHIN_DR = [
    ("kan_Knda", "tam_Taml"), ("tam_Taml", "kan_Knda"),
    ("tam_Taml", "mal_Mlym"), ("mal_Mlym", "tam_Taml"),
    ("tam_Taml", "tel_Telu"), ("tel_Telu", "tam_Taml"),
    ("tel_Telu", "mal_Mlym"), ("mal_Mlym", "tel_Telu"),
    ("kan_Knda", "mal_Mlym"), ("mal_Mlym", "kan_Knda"),
]
CROSS_IA_TO_DR = [
    ("ben_Beng", "kan_Knda"), ("hin_Deva", "kan_Knda"),
    ("hin_Deva", "mal_Mlym"), ("hin_Deva", "tam_Taml"),
    ("ory_Orya", "tel_Telu"), ("asm_Beng", "kan_Knda"),
    ("ben_Beng", "tam_Taml"), ("guj_Gujr", "mal_Mlym"),
    ("mar_Deva", "kan_Knda"), ("pan_Guru", "tam_Taml"),
]
CROSS_DR_TO_IA = [(tgt, src) for src, tgt in CROSS_IA_TO_DR]

rows = []
rows += [(s, t, "Within Indo-Aryan", "Within-family") for s, t in WITHIN_IA]
rows += [(s, t, "Within Dravidian", "Within-family") for s, t in WITHIN_DR]
rows += [(s, t, "Cross IA-to-Dravidian", "Cross-family") for s, t in CROSS_IA_TO_DR]
rows += [(s, t, "Cross Dravidian-to-IA", "Cross-family") for s, t in CROSS_DR_TO_IA]
pair_df = pd.DataFrame(rows, columns=["src_lang", "tgt_lang", "relation", "family_relation"])
pair_df["pair"] = pair_df.src_lang + "__" + pair_df.tgt_lang
assert len(pair_df) == 40 and pair_df.pair.nunique() == 40
assert pair_df.relation.value_counts().to_dict() == {
    "Within Indo-Aryan": 10, "Within Dravidian": 10,
    "Cross IA-to-Dravidian": 10, "Cross Dravidian-to-IA": 10,
}
assert set(pair_df.src_lang).union(pair_df.tgt_lang).issubset(TARGET_LANGS)
direction_set = set(map(tuple, pair_df[["src_lang", "tgt_lang"]].to_numpy()))
assert all((t, s) in direction_set for s, t in direction_set), "Every direction must have its reciprocal"
used_languages = set(pair_df.src_lang).union(pair_df.tgt_lang)
assert used_languages == TARGET_LANGS, (
    f"Pair design uses {len(used_languages)} languages, "
    f"expected {len(TARGET_LANGS)}: "
    f"missing={TARGET_LANGS - used_languages}, "
    f"extra={used_languages - TARGET_LANGS}"
)

print("Direction-balanced core:", pair_df.relation.value_counts().to_dict(), flush=True)
print("Source-language counts:\n", pair_df.src_lang.value_counts().sort_index(), flush=True)
print("Target-language counts:\n", pair_df.tgt_lang.value_counts().sort_index(), flush=True)

def get_pair_data(prow):
    """Return canonical sentence IDs plus source/target text for one direction."""
    pair = prow.pair
    frame = pd.DataFrame({
        "sentence_id": df_wide[ID_COL].to_numpy(),
        "source": df_wide[prow.src_lang].to_numpy(),
        "target": df_wide[prow.tgt_lang].to_numpy(),
    })
    assert frame.source.notna().all() and frame.target.notna().all(), f"{pair}: missing text."
    frame["run_sentence_id"] = np.arange(len(frame))
    return frame

# Freeze actual corpus identities, not just counts or positional indexes.
sample_identity = []
shared_free_corpus_ids = None
for prow in pair_df.itertuples(index=False):
    frame = get_pair_data(prow)
    assert frame.sentence_id.notna().all() and frame.sentence_id.is_unique
    selected = frame.iloc[list(FREE_SENTENCE_IDS)]
    corpus_ids = selected.sentence_id.astype(str).tolist()
    if True:
        if shared_free_corpus_ids is None:
            shared_free_corpus_ids = corpus_ids
        assert corpus_ids == shared_free_corpus_ids, f"Corpus ID mismatch for {prow.pair}"
    for row in frame.itertuples(index=False):
        sample_identity.append({
            "pair": prow.pair, "sentence_id": int(row.run_sentence_id),
            "corpus_sentence_id": str(row.sentence_id),
            "source_sha256": hashlib.sha256(str(row.source).encode()).hexdigest(),
            "target_sha256": hashlib.sha256(str(row.target).encode()).hexdigest(),
            "in_free_subset": int(row.run_sentence_id in FREE_SENTENCE_IDS),
        })
sample_identity = pd.DataFrame(sample_identity)
sample_path = os.path.join(ROOT_DIR, "manifests", "sample_identity.csv")
sample_csv = sample_identity.to_csv(index=False)
if os.path.exists(sample_path):
    with open(sample_path, encoding="utf-8", newline="") as handle:
        assert handle.read() == sample_csv, "Dataset IDs/text changed since the first run."
else:
    with open(sample_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(sample_csv)
SAMPLE_IDENTITY_SHA256 = hashlib.sha256(sample_csv.encode()).hexdigest()
if EXPECTED_INDIC_SAMPLE_IDENTITY_SHA256 is not None:
    assert SAMPLE_IDENTITY_SHA256 == EXPECTED_INDIC_SAMPLE_IDENTITY_SHA256, (
        "NLLB sample identities differ from the completed IndicTrans2 holdout."
    )
print("Sample identity SHA256:", SAMPLE_IDENTITY_SHA256)

def validate_collected_ids(frame, expected_ids, name):
    assert not frame.duplicated(["pair", "sentence_id"]).any(), f"{name}: duplicate rows"
    assert set(frame.pair) == set(pair_df.pair), f"{name}: unexpected/missing pairs"
    for prow in pair_df.itertuples(index=False):
        group = frame[frame.pair == prow.pair]
        assert set(group.sentence_id) == set(expected_ids), f"{name}/{prow.pair}: wrong ID set"
        expected = get_pair_data(prow).set_index("run_sentence_id").sentence_id.astype(str)
        actual = group.set_index("sentence_id").corpus_sentence_id.astype(str)
        assert actual.sort_index().equals(expected.loc[list(expected_ids)].sort_index()), (
            f"{name}/{prow.pair}: wrong corpus identities"
        )


# %% original code cell 9; notebook cell 11
# ============================================================
# 6. Disk, VRAM, and exact no-truncation preflight gates
# ============================================================
disk = shutil.disk_usage("/content")
free_disk_gb = disk.free / 1024**3
print(f"Free Colab disk: {free_disk_gb:.1f} GB")
if DEVICE == "cuda":
    free_vram, total_vram = torch.cuda.mem_get_info()
    free_vram_gb, total_vram_gb = free_vram / 1024**3, total_vram / 1024**3
    active_gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU VRAM: {free_vram_gb:.1f}/{total_vram_gb:.1f} GB free")
    print("Runtime lock:", {"gpu": active_gpu_name, "torch": torch.__version__,
                            "torch_cuda": torch.version.cuda})
    assert active_gpu_name == REQUIRED_GPU_NAME, (
        f"This paired run is locked to {REQUIRED_GPU_NAME}; found {active_gpu_name}."
    )
    assert torch.__version__ == REQUIRED_TORCH_VERSION, (
        f"Torch changed from {REQUIRED_TORCH_VERSION} to {torch.__version__}."
    )
    assert torch.version.cuda == REQUIRED_TORCH_CUDA, (
        f"Torch CUDA changed from {REQUIRED_TORCH_CUDA} to {torch.version.cuda}."
    )
    assert free_vram_gb >= MIN_FREE_VRAM_GB
else:
    free_vram_gb = None
    print("CPU runtime detected; the full extraction run is impractically slow.")

def encoded_sequences(texts, src_lang, tgt_lang, target=False):
    configure_languages(src_lang, tgt_lang)
    if target:
        encoded = tokenizer(text_target=texts, add_special_tokens=True, truncation=False)
    else:
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    return encoded["input_ids"]

decoder_module = model.model.decoder
ATTENTION_LAYERS = int(model.config.decoder_layers)
ATTENTION_HEADS = int(model.config.decoder_attention_heads)
assert ATTENTION_LAYERS == len(decoder_module.layers) > 0
assert ATTENTION_HEADS == decoder_module.layers[0].encoder_attn.num_heads > 0
FLOAT16_BYTES = np.dtype(np.float16).itemsize
NPY_HEADER_ALLOWANCE = 256
tf_tensor_bytes_upper = 0
free_tensor_bytes_reference = 0
free_tensor_bytes_cap_upper = 0
free_id_set = set(FREE_SENTENCE_IDS)
truncation_violations = []
max_source_tokens = 0
max_target_tokens = 0
source_unknown_tokens = 0
target_unknown_tokens = 0
unknown_token_records = []
source_content_tokens = 0
target_content_tokens = 0

for _, prow in pair_df.iterrows():
    pair_data = get_pair_data(prow)
    for start in range(0, len(pair_data), 128):
        block = pair_data.iloc[start:start + 128]
        source_sequences = encoded_sequences(block.source.astype(str).tolist(),
                                             prow.src_lang, prow.tgt_lang, False)
        target_sequences = encoded_sequences(block.target.astype(str).tolist(),
                                             prow.src_lang, prow.tgt_lang, True)
        source_lengths = [len(ids) for ids in source_sequences]
        target_lengths = [len(ids) for ids in target_sequences]
        # NLLB's SentencePiece unknown ID is an active lexical placeholder here,
        # not padding, EOS, or a language-control token. Audit and retain it.
        for local_offset, (row, source_ids, target_ids) in enumerate(
            zip(block.itertuples(index=False), source_sequences, target_sequences)
        ):
            source_unk = sum(int(x) == int(tokenizer.unk_token_id) for x in source_ids)
            target_unk = sum(int(x) == int(tokenizer.unk_token_id) for x in target_ids)
            source_content_tokens += max(0, len(source_ids) - 2)
            target_content_tokens += max(0, len(target_ids) - 2)
            for side, lang, text, ids, count in (
                ("source", prow.src_lang, row.source, source_ids, source_unk),
                ("target", prow.tgt_lang, row.target, target_ids, target_unk),
            ):
                if count:
                    unknown_token_records.append({
                        "pair": prow.pair,
                        "sentence_id": int(start + local_offset),
                        "corpus_sentence_id": str(row.sentence_id),
                        "side": side,
                        "lang": lang,
                        "unknown_count": int(count),
                        "text": str(text),
                        "tokens_json": json.dumps(token_pieces(ids), ensure_ascii=False),
                    })
        source_unknown_tokens += sum(
            int(token_id) == int(tokenizer.unk_token_id)
            for ids in source_sequences for token_id in ids
        )
        target_unknown_tokens += sum(
            int(token_id) == int(tokenizer.unk_token_id)
            for ids in target_sequences for token_id in ids
        )
        max_source_tokens = max(max_source_tokens, max(source_lengths))
        max_target_tokens = max(max_target_tokens, max(target_lengths))
        for offset, (source_length, target_length) in enumerate(zip(source_lengths, target_lengths)):
            tf_tensor_bytes_upper += (
                ATTENTION_LAYERS * ATTENTION_HEADS * source_length * target_length
                * FLOAT16_BYTES + NPY_HEADER_ALLOWANCE
            )
            run_id = start + offset
            if SAVE_FULL_TENSORS_FREE and run_id in free_id_set:
                free_tensor_bytes_reference += (
                    ATTENTION_LAYERS * ATTENTION_HEADS * source_length * target_length
                    * FLOAT16_BYTES + NPY_HEADER_ALLOWANCE
                )
                free_tensor_bytes_cap_upper += (
                    ATTENTION_LAYERS * ATTENTION_HEADS * source_length
                    * (FREE_MAX_LENGTH - 2) * FLOAT16_BYTES + NPY_HEADER_ALLOWANCE
                )
        for offset, length in enumerate(source_lengths):
            if length > MAX_LENGTH:
                truncation_violations.append((prow.pair, start + offset, "source", length))
        for offset, length in enumerate(target_lengths):
            if length > MAX_LENGTH:
                truncation_violations.append((prow.pair, start + offset, "target", length))

print("Maximum untruncated tokens:", {"source": max_source_tokens, "target": max_target_tokens})
print("Unknown-token audit:", {"source": source_unknown_tokens,
                                "target": target_unknown_tokens})
unknown_audit = pd.DataFrame(unknown_token_records, columns=[
    "pair", "sentence_id", "corpus_sentence_id", "side", "lang",
    "unknown_count", "text", "tokens_json",
])
unknown_audit_path = os.path.join(ROOT_DIR, "manifests", "unknown_token_audit.csv")
unknown_audit.to_csv(unknown_audit_path, index=False)
unknown_unique = (
    unknown_audit.drop_duplicates(["corpus_sentence_id", "side", "lang", "text"])
    if len(unknown_audit) else unknown_audit.copy()
)
unknown_unique_path = os.path.join(
    ROOT_DIR, "manifests", "unknown_token_audit_unique.csv"
)
unknown_unique.to_csv(unknown_unique_path, index=False)
if source_unknown_tokens or target_unknown_tokens:
    print(
        "WARNING: <unk> is retained as a lexical placeholder; inspect",
        unknown_unique_path,
    )
    display(unknown_unique[[
        "corpus_sentence_id", "side", "lang", "unknown_count", "text"
    ]])
tensor_reference_gib = (tf_tensor_bytes_upper + free_tensor_bytes_reference) / 1024**3
tensor_cap_upper_gib = (tf_tensor_bytes_upper + free_tensor_bytes_cap_upper) / 1024**3
required_disk_gib = tensor_cap_upper_gib * DISK_SAFETY_FACTOR + DISK_FIXED_OVERHEAD_GB
print("Tensor storage estimate (GiB):", {
    "reference_length_estimate": round(tensor_reference_gib, 2),
    "generation_cap_upper": round(tensor_cap_upper_gib, 2),
    "required_with_margin": round(required_disk_gib, 2),
})
assert free_disk_gb >= required_disk_gib, (
    f"Need about {required_disk_gib:.1f} GiB free; found {free_disk_gb:.1f} GiB."
)
assert not truncation_violations, (
    f"Found {len(truncation_violations)} sequences over MAX_LENGTH={MAX_LENGTH}; "
    f"first: {truncation_violations[:10]}"
)
assert int(tokenizer.unk_token_id) not in {
    int(tokenizer.pad_token_id), int(tokenizer.eos_token_id)
}, "NLLB unknown ID unexpectedly aliases a control token."
assert int(unknown_audit.unknown_count.sum()) == (
    source_unknown_tokens + target_unknown_tokens
), "Unknown-token audit is incomplete."
PREFLIGHT_PASSED = True
PREFLIGHT_REPORT = {
    "free_disk_gib": free_disk_gb, "free_vram_gib": free_vram_gb,
    "max_source_tokens": max_source_tokens, "max_target_tokens": max_target_tokens,
    "source_unknown_tokens": source_unknown_tokens,
    "target_unknown_tokens": target_unknown_tokens,
    "source_unknown_rate": source_unknown_tokens / max(1, source_content_tokens),
    "target_unknown_rate": target_unknown_tokens / max(1, target_content_tokens),
    "unique_unknown_records": len(unknown_unique),
    "unknown_token_policy": (
        "retain as active lexical placeholder; exclude only language/pad/EOS controls; "
        "record all corpus occurrences"
    ),
    "truncation_violations": len(truncation_violations),
    "attention_layers": ATTENTION_LAYERS, "attention_heads": ATTENTION_HEADS,
    "tf_tensor_gib_upper": tf_tensor_bytes_upper / 1024**3,
    "free_tensor_gib_reference_estimate": free_tensor_bytes_reference / 1024**3,
    "free_tensor_gib_cap_upper": free_tensor_bytes_cap_upper / 1024**3,
    "required_disk_gib_with_margin": required_disk_gib,
    "disk_safety_factor": DISK_SAFETY_FACTOR,
    "fixed_overhead_gib": DISK_FIXED_OVERHEAD_GB,
}
with open(os.path.join(ROOT_DIR, "manifests", "preflight.json"), "w") as handle:
    json.dump(PREFLIGHT_REPORT, handle, indent=2)
print("PRE-FLIGHT PASSED: disk, VRAM, NLLB language coverage and no truncation.")


# %% original code cell 10; notebook cell 12
# ============================================================
# 6b. Exact NLLB source/target preparation
# ============================================================
def prepare_pair(src_lang, tgt_lang, src_text, tgt_text):
    configure_languages(src_lang, tgt_lang)
    src_tok = tokenizer(
        [str(src_text)], return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_LENGTH, return_attention_mask=True,
    )
    tgt_tok = tokenizer(
        text_target=[str(tgt_text)], return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH, add_special_tokens=True,
        return_attention_mask=True,
    )
    labels = tgt_tok["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    tgt_tok["labels"] = labels
    return str(src_text), str(tgt_text), src_tok, tgt_tok


def shift_labels_right(labels):
    """Exact M2M100/NLLB decoder shift; replaces unavailable convenience APIs."""
    pad_id = int(model.config.pad_token_id)
    start_id = int(model.config.decoder_start_token_id)
    shifted = labels.new_full(labels.shape, pad_id)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = start_id
    shifted.masked_fill_(shifted == -100, pad_id)
    return shifted


# %% original code cell 11; notebook cell 13
# ============================================================
# 7. Real-sentence NLLB tokenization and row-label sanity check
# ============================================================
demo = pair_df.iloc[0]
demo_data = get_pair_data(demo).iloc[0]
src_text0, tgt_text0, src_tok0, tgt_tok0 = prepare_pair(
    demo.src_lang, demo.tgt_lang, demo_data.source, demo_data.target
)
src_ids0 = src_tok0["input_ids"][0].tolist()
tgt_ids0 = tgt_tok0["input_ids"][0].tolist()
assert src_ids0[0] == language_id(demo.src_lang) and src_ids0[-1] == tokenizer.eos_token_id
assert tgt_ids0[0] == language_id(demo.tgt_lang) and tgt_ids0[-1] == tokenizer.eos_token_id
decoder_ids0 = shift_labels_right(tgt_tok0["labels"].to(DEVICE))[0].cpu()
assert len(decoder_ids0) == len(tgt_ids0)
assert int(decoder_ids0[0]) == int(model.config.decoder_start_token_id)
assert torch.equal(decoder_ids0[1:], tgt_tok0["labels"][0, :-1].cpu())
assert _probe_baseline == tokenizer_state_probe("after prepare_pair")
print("Source controls:", token_pieces([src_ids0[0], src_ids0[-1]]))
print("Target controls:", token_pieces([tgt_ids0[0], tgt_ids0[-1]]))
print("Teacher-forced attention rows map to the predicted label tokens.")


# %% original code cell 12; notebook cell 15
# ============================================================
# 8. fp16 vs fp32 sensitivity check (small subsample)
# ============================================================
def get_cross_attention_mean(model_, src_tok, tgt_tok):
    with torch.no_grad():
        out = model_(
            input_ids=src_tok["input_ids"].to(model_.device),
            attention_mask=src_tok["attention_mask"].to(model_.device),
            labels=tgt_tok["input_ids"].to(model_.device),
            output_attentions=True,
            return_dict=True,
        )
    attn = torch.stack(out.cross_attentions, dim=0)   # [L, B, H, T, S]
    return attn[:, 0].float().cpu().numpy()           # [L, H, T, S]

N_FP_CHECK = min(25, N_SENTENCES)
diffs = []
FP16_FP32_MEAN_MAX_DIFF = None
FP16_FP32_OVERALL_MAX_DIFF = None
model_fp32 = None
if DEVICE == "cuda":
    print(f"Loading a temporary fp32 copy for a {N_FP_CHECK}-sentence comparison...")
    model_fp32 = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        attn_implementation="eager", token=HF_TOKEN, revision=RESOLVED_MODEL_COMMIT,
    ).to(DEVICE).eval()

    src_l0, tgt_l0 = pair_df.iloc[0]["src_lang"], pair_df.iloc[0]["tgt_lang"]
    fp_pair_data = get_pair_data(pair_df.iloc[0])
    for i in range(N_FP_CHECK):
        r = fp_pair_data.iloc[i]
        _, _, s_tok, t_tok = prepare_pair(src_l0, tgt_l0, r.source, r.target)
        a16 = get_cross_attention_mean(model, s_tok, t_tok)
        a32 = get_cross_attention_mean(model_fp32, s_tok, t_tok)
        diffs.append(np.abs(a16 - a32).max())

    print(f"Max abs attention-weight difference fp16 vs fp32 over {N_FP_CHECK} sentences:")
    print(f"  mean of per-sentence max-diff: {np.mean(diffs):.6f}")
    FP16_FP32_MEAN_MAX_DIFF = float(np.mean(diffs))
    FP16_FP32_OVERALL_MAX_DIFF = float(np.max(diffs))
    print(f"  overall max-diff:              {FP16_FP32_OVERALL_MAX_DIFF:.6f}")
    del model_fp32
    gc.collect(); torch.cuda.empty_cache()
else:
    print("No GPU available in this environment -- skipping fp16/fp32 check "
          "(run on the actual GPU environment before trusting fp16 results).")


# %% original code cell 13; notebook cell 17
# ============================================================
# 9. Determinism check
# ============================================================
src_l0, tgt_l0 = pair_df.iloc[0]["src_lang"], pair_df.iloc[0]["tgt_lang"]
r0 = get_pair_data(pair_df.iloc[0]).iloc[0]
_, _, s_tok, t_tok = prepare_pair(src_l0, tgt_l0, r0.source, r0.target)

a_run1 = get_cross_attention_mean(model, s_tok, t_tok)
a_run2 = get_cross_attention_mean(model, s_tok, t_tok)
max_diff = np.abs(a_run1 - a_run2).max()
REPEAT_RUN_MAX_DIFF = float(max_diff)
BIT_IDENTICAL = bool(max_diff == 0)
print(f"Max abs difference between two identical repeated runs: {max_diff:.8f}")
if max_diff > 0:
    print("Nonzero -- this pipeline is not bit-deterministic on this GPU/driver combo. "
          "Set STRICT_DETERMINISM = True above and re-run this cell if you need "
          "run-to-run reproducibility at this precision; note it will be slower.")
else:
    print("Bit-identical across repeated runs.")

# %% original code cell 14; notebook cell 18
# ============================================================
# 10. Attention summary definitions
# ============================================================
def summarize_matrix(mat):
    """Summarize a [target, source] map after row renormalization."""
    if mat.ndim != 2 or min(mat.shape) == 0:
        raise ValueError(f"Expected a non-empty 2D matrix, got {mat.shape}")
    finite = mat[np.isfinite(mat)]
    p = mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-12, None)
    t_len, s_len = p.shape
    t_pos = np.arange(t_len)[:, None] / max(t_len - 1, 1)
    s_pos = np.arange(s_len)[None, :] / max(s_len - 1, 1)
    displacement = np.abs(t_pos - s_pos)
    result = {
        "mean": float(finite.mean()) if finite.size else np.nan,
        "std": float(finite.std()) if finite.size else np.nan,
        "min": float(finite.min()) if finite.size else np.nan,
        "max": float(finite.max()) if finite.size else np.nan,
        "mean_row_max": float(p.max(axis=1).mean()),
        "row_max_baseline": 1.0 / s_len,
        "normalized_entropy": float(np.mean(
            -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1) / np.log(max(s_len, 2))
        )),
        "expected_normalized_displacement": float((p * displacement).sum() / t_len),
        "src_len": int(s_len),
        "tgt_len": int(t_len),
    }
    result["row_max_excess"] = result["mean_row_max"] - result["row_max_baseline"]
    for pct in (5, 10, 20):
        band = (displacement <= pct / 100.0).astype(float)
        mass = float((p * band).sum() / t_len)
        baseline = float(band.sum() / (t_len * s_len))
        result[f"diagonal_mass_{pct}pct"] = mass
        result[f"uniform_diagonal_baseline_{pct}pct"] = baseline
        result[f"diagonal_excess_{pct}pct"] = mass - baseline
    # Compatibility alias; new analyses must name the band explicitly.
    result["diagonal_excess"] = result["diagonal_excess_10pct"]
    return result


# %% original code cell 15; notebook cell 20
# ============================================================
# 11b. NLLB extraction: padding, control-token masking and row alignment
# ============================================================
def _active_positions(input_ids, attention_mask):
    assert len(input_ids) == len(attention_mask)
    return [i for i, m in enumerate(attention_mask) if int(m) == 1]

def _is_punctuation_piece(piece):
    text = str(piece).replace("▁", "").replace("Ġ", "").strip()
    if not text:
        return True
    return all(unicodedata.category(ch)[0] in {"P", "S", "Z", "C"} for ch in text)

def _kept_source_positions(input_ids, attention_mask, src_lang):
    """Remove exactly the NLLB source-language prefix and terminal EOS."""
    active = _active_positions(input_ids, attention_mask)
    assert len(active) >= 3
    assert int(input_ids[active[0]]) == language_id(src_lang)
    assert int(input_ids[active[-1]]) == int(tokenizer.eos_token_id)
    controls = [active[0], active[-1]]
    kept = active[1:-1]
    assert all(int(input_ids[i]) != int(tokenizer.pad_token_id) for i in kept)
    return kept, [active[0]], controls

def _kept_label_positions(input_ids, attention_mask, tgt_lang):
    """Rows are labelled by predicted tokens: [tgt tag], lexical..., [EOS]."""
    active = _active_positions(input_ids, attention_mask)
    assert len(active) >= 3
    assert int(input_ids[active[0]]) == language_id(tgt_lang)
    assert int(input_ids[active[-1]]) == int(tokenizer.eos_token_id)
    return active[1:-1], [active[0], active[-1]]

def _kept_generated_positions(generated_ids, tgt_lang):
    assert generated_ids and int(generated_ids[0]) == language_id(tgt_lang)
    eos_ids = _eos_ids()
    controls = [0]
    stop = len(generated_ids)
    if int(generated_ids[-1]) in eos_ids:
        controls.append(len(generated_ids) - 1)
        stop -= 1
    return list(range(1, stop)), controls

def _prepare_teacher_forced_batch(src_lang, tgt_lang, src_texts, tgt_texts):
    configure_languages(src_lang, tgt_lang)
    src_tok = tokenizer(
        [str(x) for x in src_texts], return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH, return_attention_mask=True,
    )
    tgt_tok = tokenizer(
        text_target=[str(x) for x in tgt_texts], return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH, add_special_tokens=True,
        return_attention_mask=True,
    )
    return src_tok, tgt_tok

def _layer_summaries(full, src_lex, tgt_lex):
    result = {
        "layer_diagonal_excess_10pct_json": json.dumps([]),
        "layer_expected_normalized_displacement_json": json.dumps([]),
    }
    if not src_lex or not tgt_lex:
        return result
    lexical = full[:, :, tgt_lex, :][:, :, :, src_lex]
    layer_diag, layer_disp = [], []
    for layer in range(lexical.shape[0]):
        stats = summarize_matrix(lexical[layer].mean(axis=0))
        layer_diag.append(stats["diagonal_excess_10pct"])
        layer_disp.append(stats["expected_normalized_displacement"])
    result["layer_diagonal_excess_10pct_json"] = json.dumps(layer_diag)
    result["layer_expected_normalized_displacement_json"] = json.dumps(layer_disp)
    return result

def _summaries_with_token_controls(full, source_ids, target_ids,
                                   source_tag_positions, source_control_positions,
                                   target_control_positions, target_row_alignment):
    # <unk> represents an active lexical span. Although Hugging Face includes it
    # in all_special_ids, it must not be conflated with language/pad/EOS controls.
    _disallowed_controls = (
        set(int(x) for x in tokenizer.all_special_ids)
        - {int(tokenizer.unk_token_id)}
    )
    assert not _disallowed_controls.intersection(int(x) for x in source_ids), (
        "An NLLB source control token survived control masking."
    )
    assert not _disallowed_controls.intersection(int(x) for x in target_ids), (
        "An NLLB target control token survived control masking."
    )
    source_tokens = token_pieces(source_ids)
    target_tokens = token_pieces(target_ids)
    all_map = full.mean(axis=(0, 1))
    src_lex = [i for i, piece in enumerate(source_tokens) if not _is_punctuation_piece(piece)]
    tgt_lex = [i for i, piece in enumerate(target_tokens) if not _is_punctuation_piece(piece)]
    all_stats = summarize_matrix(all_map)
    lexical_available = bool(src_lex and tgt_lex)
    if lexical_available:
        primary = summarize_matrix(all_map[np.ix_(tgt_lex, src_lex)])
        primary["analysis_view"] = "lexical_no_punctuation"
    else:
        primary = {key: np.nan for key in all_stats}
        primary.update({"src_len": len(src_lex), "tgt_len": len(tgt_lex),
                        "analysis_view": "lexical_no_punctuation_unavailable"})
    for key, value in all_stats.items():
        primary[f"all_retained_{key}"] = value
    primary.update(_layer_summaries(full, src_lex, tgt_lex))
    primary.update({
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "target_row_alignment": target_row_alignment,
        "source_tags_removed": 1,
        "source_special_tokens_removed": len(source_control_positions),
        "target_special_tokens_removed": len(target_control_positions),
        "source_tag_positions_json": json.dumps(source_tag_positions),
        "source_special_positions_json": json.dumps(source_control_positions),
        "target_special_positions_json": json.dumps(target_control_positions),
        "src_retained_tokens": len(source_tokens),
        "tgt_retained_tokens": len(target_tokens),
        "source_unknown_tokens": sum(
            int(x) == int(tokenizer.unk_token_id) for x in source_ids
        ),
        "target_unknown_tokens": sum(
            int(x) == int(tokenizer.unk_token_id) for x in target_ids
        ),
        "src_lexical_tokens": len(src_lex), "tgt_lexical_tokens": len(tgt_lex),
        "src_punctuation_tokens": len(source_tokens) - len(src_lex),
        "tgt_punctuation_tokens": len(target_tokens) - len(tgt_lex),
        "source_token_ids_json": json.dumps([int(x) for x in source_ids]),
        "target_token_ids_json": json.dumps([int(x) for x in target_ids]),
        "source_tokens_json": json.dumps(source_tokens, ensure_ascii=False),
        "target_tokens_json": json.dumps(target_tokens, ensure_ascii=False),
    })
    return primary

def extract_attention_teacher_forced_batch(src_lang, tgt_lang, src_texts, tgt_texts):
    src_tok, tgt_tok = _prepare_teacher_forced_batch(src_lang, tgt_lang, src_texts, tgt_texts)
    labels = tgt_tok["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels_device = labels.to(DEVICE)
    decoder_ids = shift_labels_right(labels_device)
    assert decoder_ids.shape == labels.shape
    with torch.inference_mode():
        out = model(
            input_ids=src_tok["input_ids"].to(DEVICE),
            attention_mask=src_tok["attention_mask"].to(DEVICE),
            decoder_input_ids=decoder_ids, labels=labels_device,
            output_attentions=True, return_dict=True,
        )
    stacked = torch.stack([x.float().cpu() for x in out.cross_attentions], dim=0).numpy()
    assert stacked.shape[3] == labels.shape[1]
    del out, decoder_ids
    results = []
    for b in range(len(src_texts)):
        source_ids = src_tok["input_ids"][b].tolist()
        source_mask = src_tok["attention_mask"][b].tolist()
        target_ids = tgt_tok["input_ids"][b].tolist()
        target_mask = tgt_tok["attention_mask"][b].tolist()
        src_keep, tag_positions, src_controls = _kept_source_positions(
            source_ids, source_mask, src_lang
        )
        tgt_keep, tgt_controls = _kept_label_positions(target_ids, target_mask, tgt_lang)
        if not src_keep or not tgt_keep:
            results.append(None)
            continue
        full = stacked[:, b][:, :, tgt_keep, :][:, :, :, src_keep]
        kept_src_ids = [source_ids[i] for i in src_keep]
        kept_tgt_ids = [target_ids[i] for i in tgt_keep]
        stat = _summaries_with_token_controls(
            full, kept_src_ids, kept_tgt_ids,
            tag_positions, src_controls, tgt_controls, "predicted_label_token",
        )
        results.append((full, stat))
    return results

def _eos_ids():
    values = [tokenizer.eos_token_id, model.generation_config.eos_token_id]
    result = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(x) for x in value)
        else:
            result.add(int(value))
    return result

def extract_attention_free_batch(src_lang, tgt_lang, src_texts, max_length=None):
    # Resolve at call time so later variable changes cannot leave a stale Python default.
    if max_length is None:
        max_length = int(FREE_MAX_LENGTH)
    max_length = int(max_length)
    assert max_length == int(FREE_MAX_LENGTH)
    configure_languages(src_lang, tgt_lang)
    src_tok = tokenizer(
        [str(x) for x in src_texts], return_tensors="pt", padding=True,
        truncation=True, max_length=MAX_LENGTH, return_attention_mask=True,
    )
    with torch.inference_mode():
        generated = model.generate(
            input_ids=src_tok["input_ids"].to(DEVICE),
            attention_mask=src_tok["attention_mask"].to(DEVICE),
            forced_bos_token_id=language_id(tgt_lang),
            num_beams=GREEDY_NUM_BEAMS, num_return_sequences=1,
            min_length=0, do_sample=DO_SAMPLE, use_cache=True,
            max_length=max_length, output_attentions=True,
            return_dict_in_generate=True,
        )
    if not getattr(generated, "cross_attentions", None):
        raise RuntimeError("NLLB generation returned no cross-attention tensors.")
    n_steps = len(generated.cross_attentions)
    n_layers = len(generated.cross_attentions[0])
    layers = [
        torch.cat([generated.cross_attentions[s][layer].float().cpu()
                   for s in range(n_steps)], dim=2)
        for layer in range(n_layers)
    ]
    stacked = torch.stack(layers, dim=0).numpy()
    eos_ids = _eos_ids()
    decoded_ids, bookkeeping = [], []
    for b in range(len(src_texts)):
        sequence = generated.sequences[b].detach().cpu().tolist()
        prefix_len = len(sequence) - n_steps
        assert prefix_len == 1, ("Unexpected NLLB decoder prefix", len(sequence), n_steps)
        predicted_ids = [int(x) for x in sequence[prefix_len:]]
        assert predicted_ids[0] == language_id(tgt_lang)
        active_steps = n_steps
        for j, token_id in enumerate(predicted_ids):
            if token_id in eos_ids:
                active_steps = j + 1
                break
        active_ids = predicted_ids[:active_steps]
        decoded_ids.append(sequence[:prefix_len] + active_ids)
        bookkeeping.append((active_ids, active_steps, prefix_len))
    hypotheses = tokenizer.batch_decode(
        decoded_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert len(hypotheses) == len(src_texts)
    results = []
    for b, hypothesis in enumerate(hypotheses):
        active_ids, active_steps, prefix_len = bookkeeping[b]
        source_ids = src_tok["input_ids"][b].tolist()
        source_mask = src_tok["attention_mask"][b].tolist()
        src_keep, tag_positions, src_controls = _kept_source_positions(
            source_ids, source_mask, src_lang
        )
        tgt_keep, tgt_controls = _kept_generated_positions(active_ids, tgt_lang)
        if not src_keep or not tgt_keep:
            results.append(None)
            continue
        full = stacked[:, b][:, :, tgt_keep, :][:, :, :, src_keep]
        kept_src_ids = [source_ids[i] for i in src_keep]
        kept_tgt_ids = [active_ids[i] for i in tgt_keep]
        stat = _summaries_with_token_controls(
            full, kept_src_ids, kept_tgt_ids,
            tag_positions, src_controls, tgt_controls, "predicted_generated_token",
        )
        ended_with_eos = int(bool(active_ids) and active_ids[-1] in eos_ids)
        reached_length_boundary = int(
            not ended_with_eos and (prefix_len + active_steps) >= max_length
        )
        stat.update({
            "generated_steps": int(active_steps),
            "generation_ended_with_eos": ended_with_eos,
            "generation_hit_max_length": reached_length_boundary,
        })
        results.append((full, stat, hypothesis))
    del generated
    return results

def probe_language_controls(prow):
    """Verify documented default-mode source and target layouts for a real row."""
    row = get_pair_data(prow).iloc[0]
    configure_languages(prow.src_lang, prow.tgt_lang)

    source_encoded = tokenizer(
        str(row.source), add_special_tokens=True, return_attention_mask=True
    )
    source_ids = source_encoded["input_ids"]
    source_mask = source_encoded["attention_mask"]
    source_keep, source_tag_positions, source_control_positions = (
        _kept_source_positions(source_ids, source_mask, prow.src_lang)
    )
    assert token_pieces([source_ids[i] for i in source_tag_positions]) == [prow.src_lang]
    assert token_pieces([source_ids[i] for i in source_control_positions]) == [
        prow.src_lang, "</s>"
    ]
    assert prow.tgt_lang not in token_pieces([source_ids[i] for i in source_keep])

    target_encoded = tokenizer(
        text_target=[str(row.target)], add_special_tokens=True,
        return_attention_mask=True,
    )
    target_ids = target_encoded["input_ids"][0]
    target_mask = target_encoded["attention_mask"][0]
    target_keep, target_control_positions = _kept_label_positions(
        target_ids, target_mask, prow.tgt_lang
    )
    assert token_pieces([target_ids[i] for i in target_control_positions]) == [
        prow.tgt_lang, "</s>"
    ]
    assert prow.src_lang not in token_pieces([target_ids[i] for i in target_keep])

    return {
        "pair": prow.pair,
        "source_tag": prow.src_lang,
        "target_tag": prow.tgt_lang,
        "source_control_positions": source_control_positions,
        "target_control_positions": target_control_positions,
    }

PREFIX_PROBE_RECORDS = [
    probe_language_controls(prow) for _, prow in pair_df.iterrows()
]
PREFIX_ASSUMPTION_PASSED = len(PREFIX_PROBE_RECORDS) == len(pair_df)
print("Verified default-mode NLLB source and target controls for all 40 directions.")

seed_everything(SEED)
_smoke = pair_df.iloc[0]
_smoke_data = get_pair_data(_smoke).iloc[:2]
_srcs, _tgts = _smoke_data.source.tolist(), _smoke_data.target.tolist()
_batch_src, _batch_tgt = _prepare_teacher_forced_batch(
    _smoke.src_lang, _smoke.tgt_lang, _srcs, _tgts
)
for b, (src, tgt) in enumerate(zip(_srcs, _tgts)):
    _, _, single_src, single_tgt = prepare_pair(_smoke.src_lang, _smoke.tgt_lang, src, tgt)
    for label, batch_tok, single_tok in (
        ("source", _batch_src, single_src), ("target", _batch_tgt, single_tgt)
    ):
        batched = batch_tok["input_ids"][b][batch_tok["attention_mask"][b].bool()]
        single = single_tok["input_ids"][0][single_tok["attention_mask"][0].bool()]
        assert torch.equal(batched, single), f"Batch/single {label} IDs differ for item {b}"
_tf_smoke = extract_attention_teacher_forced_batch(
    _smoke.src_lang, _smoke.tgt_lang, _srcs, _tgts
)
assert all(r is not None and r[0].ndim == 4 and np.isfinite(r[0]).all() for r in _tf_smoke)

# Exercise the exact production TF batch size and compare two boundary items
# against independent single-item calls. This checks numerical attention results,
# not merely padded token IDs.
_batch_probe_frame = df_wide_all.iloc[
    CORPUS_ROW_START:CORPUS_ROW_START + BATCH_SIZE_TF
].reset_index(drop=True)
assert len(_batch_probe_frame) == BATCH_SIZE_TF
_probe_srcs = _batch_probe_frame[_smoke.src_lang].astype(str).tolist()
_probe_tgts = _batch_probe_frame[_smoke.tgt_lang].astype(str).tolist()
_batch32_results = extract_attention_teacher_forced_batch(
    _smoke.src_lang, _smoke.tgt_lang, _probe_srcs, _probe_tgts
)
assert len(_batch32_results) == BATCH_SIZE_TF and all(x is not None for x in _batch32_results)
_raw_diffs, _summary_diffs = [], []
_batch_invariance_metrics = [
    "diagonal_excess_10pct", "expected_normalized_displacement",
    "normalized_entropy", "mean_row_max",
]
for _idx in (0, BATCH_SIZE_TF - 1):
    _single_result = extract_attention_teacher_forced_batch(
        _smoke.src_lang, _smoke.tgt_lang,
        [_probe_srcs[_idx]], [_probe_tgts[_idx]],
    )[0]
    assert _single_result is not None
    _batch_tensor, _batch_stat = _batch32_results[_idx]
    _single_tensor, _single_stat = _single_result
    assert _batch_tensor.shape == _single_tensor.shape
    _raw_diffs.append(float(np.max(np.abs(_batch_tensor - _single_tensor))))
    _summary_diffs.extend(
        abs(float(_batch_stat[key]) - float(_single_stat[key]))
        for key in _batch_invariance_metrics
    )
TF_BATCH_ATTENTION_MAX_ABS_DIFF = max(_raw_diffs)
TF_BATCH_SUMMARY_MAX_ABS_DIFF = max(_summary_diffs)
assert TF_BATCH_ATTENTION_MAX_ABS_DIFF <= 0.01, (
    f"Batch-32 raw attention difference too large: {TF_BATCH_ATTENTION_MAX_ABS_DIFF}"
)
assert TF_BATCH_SUMMARY_MAX_ABS_DIFF <= 0.001, (
    f"Batch-32 summary difference too large: {TF_BATCH_SUMMARY_MAX_ABS_DIFF}"
)
print("TF batch-32 numerical invariance:", {
    "raw_attention_max_abs_diff": TF_BATCH_ATTENTION_MAX_ABS_DIFF,
    "summary_max_abs_diff": TF_BATCH_SUMMARY_MAX_ABS_DIFF,
})

_free_smoke = extract_attention_free_batch(
    _smoke.src_lang, _smoke.tgt_lang, [_srcs[0]], max_length=FREE_MAX_LENGTH
)[0]
assert _free_smoke is not None and _free_smoke[0].ndim == 4
assert np.isfinite(_free_smoke[0]).all() and _free_smoke[2].strip()
assert _free_smoke[1]["generation_hit_max_length"] == 0
assert _free_smoke[1]["source_tags_removed"] == 1
assert _free_smoke[1]["source_special_tokens_removed"] == 2
FREE_SMOKE_GENERATION_CAP_HITS = int(_free_smoke[1]["generation_hit_max_length"])
EXTRACTION_SMOKE_PASSED = True
TF_BATCH_EQUIVALENCE_PASSED = True
EDGE_CASE_SMOKE_REPORT = []
print("Production smoke passed: NLLB tags, predicted-token rows, batching and greedy attention.")

timing_n = 8
timing_pairs = np.linspace(0, len(pair_df) - 1, timing_n, dtype=int)
timing_sentences = np.linspace(0, N_SENTENCES - 1, timing_n, dtype=int)
timing_start = time.perf_counter()
with tempfile.TemporaryDirectory() as timing_dir:
    for j, (pair_index, sentence_id) in enumerate(zip(timing_pairs, timing_sentences)):
        prow = pair_df.iloc[int(pair_index)]
        text = get_pair_data(prow).iloc[int(sentence_id)].source
        seed_everything(SEED + int(pair_index))
        result = extract_attention_free_batch(
            prow.src_lang, prow.tgt_lang, [text], max_length=FREE_MAX_LENGTH
        )[0]
        if result is not None:
            np.save(os.path.join(timing_dir, f"{j}.npy"), result[0].astype(np.float16))
timing_elapsed = time.perf_counter() - timing_start
seconds_per_free_item = timing_elapsed / timing_n
projected_free_hours = seconds_per_free_item * len(pair_df) * len(FREE_SENTENCE_IDS) / 3600
FREE_TIMING_REPORT = {
    "probe_items": timing_n, "elapsed_seconds": timing_elapsed,
    "seconds_per_item": seconds_per_free_item,
    "projected_free_hours_compute_and_write": projected_free_hours,
    "projected_free_hours_with_15pct_checkpoint_allowance": projected_free_hours * 1.15,
}
print("Free-generation timing projection:", FREE_TIMING_REPORT)
del _tf_smoke, _free_smoke, _batch_src, _batch_tgt, _batch32_results
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %% original code cell 16; notebook cell 22
# Reproducibility manifest written before collection and checked on resume
package_names = ["transformers", "tokenizers", "datasets", "huggingface-hub", "torch",
                 "sacrebleu", "numpy", "pandas", "pyarrow", "sentencepiece"]
package_versions = {}
for package in package_names:
    try:
        package_versions[package] = version(package)
    except PackageNotFoundError:
        package_versions[package] = None
run_manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
    "pipeline_source_sha256": PIPELINE_SOURCE_SHA256,
    "config": CORE_CONFIG, "source_lock": SOURCE_LOCK,
    "resolved_model_commit": RESOLVED_MODEL_COMMIT,
    "resolved_dataset_commit": RESOLVED_DATASET_COMMIT,
    "dataset_fingerprint": DATASET_FINGERPRINT,
    "sample_identity_sha256": SAMPLE_IDENTITY_SHA256,
    "expected_indic_sample_identity_sha256": EXPECTED_INDIC_SAMPLE_IDENTITY_SHA256,
    "packages": package_versions, "python": platform.python_version(),
    "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "gpu_total_bytes": (torch.cuda.get_device_properties(0).total_memory
                        if DEVICE == "cuda" else None),
    "dtype": str(DTYPE), "pair_list": pair_df.to_dict("records"),
    "preflight": PREFLIGHT_REPORT,
    "validation_observations": {
        "fp16_fp32_mean_max_diff": FP16_FP32_MEAN_MAX_DIFF,
        "fp16_fp32_overall_max_diff": FP16_FP32_OVERALL_MAX_DIFF,
        "repeat_run_max_diff": REPEAT_RUN_MAX_DIFF,
        "bit_identical": BIT_IDENTICAL,
        "nllb_source_prefix_verified": PREFIX_ASSUMPTION_PASSED,
        "tf_batch_equivalence_passed": TF_BATCH_EQUIVALENCE_PASSED,
        "tf_batch_attention_max_abs_diff": TF_BATCH_ATTENTION_MAX_ABS_DIFF,
        "tf_batch_summary_max_abs_diff": TF_BATCH_SUMMARY_MAX_ABS_DIFF,
        "free_smoke_generation_cap_hits": FREE_SMOKE_GENERATION_CAP_HITS,
    },
    "free_timing_probe": FREE_TIMING_REPORT,
    "processor_protocol": "raw FLORES text; NLLB default non-legacy language conditioning",
    "token_vocabulary_protocol": "single NLLB SentencePiece vocabulary",
    "attention_mask_protocol": (
        "remove source language prefix and EOS; target rows use predicted label/generated "
        "tokens with target language prefix and EOS removed"
    ),
    "metric_protocol": "FLORES-200 spBLEU; raw-text chrF order 6 beta 2; chrF++ word order 2",
    "free_subset_policy": "fixed positional IDs; no substitution; strict unresolved-failure gate",
    "shared_content_across_pairs": True,
    "runtime_gates_passed": False,
}
run_manifest_path = os.path.join(ROOT_DIR, "manifests", "run_manifest.json")
if os.path.isfile(run_manifest_path):
    with open(run_manifest_path) as handle:
        previous = json.load(handle)
    for key in ("config", "source_lock", "sample_identity_sha256",
                "pipeline_schema_version", "pipeline_source_sha256", "packages",
                "python", "torch_cuda", "dtype"):
        assert previous[key] == run_manifest[key], f"Resume provenance changed: {key}"
else:
    with open(run_manifest_path, "w") as handle:
        json.dump(run_manifest, handle, indent=2)
sessions = os.path.join(ROOT_DIR, "manifests", "runtime_sessions.jsonl")
with open(sessions, "a") as handle:
    handle.write(json.dumps(run_manifest) + "\n")
print("Pre-run manifest and session provenance saved.")


# %% original code cell 17; notebook cell 23
# ============================================================
# 12. Main collection loop with resume and capped-output handling
# ============================================================

assert FREE_MAX_LENGTH == 256
assert BEAM_MAX_LENGTH == 256

tf_dir = os.path.join(ROOT_DIR, "teacher_forced")
free_dir = os.path.join(ROOT_DIR, "free_generation")
free_capped_dir = os.path.join(ROOT_DIR, "free_generation_capped")
cap_audit_path = os.path.join(ROOT_DIR, "manifests", "generation_cap_audit.csv")
log_path = os.path.join(ROOT_DIR, "logs", "failures.csv")
repair_path = os.path.join(ROOT_DIR, "logs", "checkpoint_repair_audit.csv")
N_FREE = len(FREE_SENTENCE_IDS)


def _atomic_to_csv(data, path):
    temporary = path + ".tmp"
    data.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_save_npy(array, path):
    temporary = path + ".tmp.npy"
    np.save(temporary, array)
    os.replace(temporary, path)


def _write_checkpoint(data, path):
    clean = (
        data.drop_duplicates("sentence_id", keep="last")
        .sort_values("sentence_id")
    )
    _atomic_to_csv(clean, path)


def _tensor_is_valid(path):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) <= 128:
            return False

        array = np.load(path, mmap_mode="r", allow_pickle=False)

        return (
            array.ndim == 4
            and all(int(x) > 0 for x in array.shape)
        )
    except Exception:
        return False


REPAIR_COLUMNS = [
    "utc",
    "pair",
    "mode",
    "sentence_id",
    "reason",
    "checkpoint",
]

if os.path.isfile(repair_path) and os.path.getsize(repair_path):
    checkpoint_repairs = pd.read_csv(repair_path).to_dict("records")
else:
    checkpoint_repairs = []


def _record_repairs(rows):
    if not rows:
        return

    now = datetime.now(timezone.utc).isoformat()

    for row in rows:
        checkpoint_repairs.append({
            "utc": now,
            **row,
        })

    repair_frame = pd.DataFrame(
        checkpoint_repairs,
        columns=REPAIR_COLUMNS,
    )
    _atomic_to_csv(repair_frame, repair_path)


COMMON_REQUIRED = {
    "pair",
    "sentence_id",
    "corpus_sentence_id",
    "pipeline_schema_version",
    "source_tags_removed",
    "source_special_tokens_removed",
    "target_row_alignment",
    "source_token_ids_json",
    "target_token_ids_json",
    "source_tokens_json",
    "target_tokens_json",
    "diagonal_excess_10pct",
    "expected_normalized_displacement",
}


def _load_checkpoint(
    path,
    tensor_dir,
    pair_name,
    mode,
    expected_ids,
    require_tensor,
):
    if not (RESUME and os.path.isfile(path)):
        return pd.DataFrame()

    try:
        data = pd.read_csv(
            path,
            dtype={"corpus_sentence_id": str},
        )
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()

    repairs = []

    if not COMMON_REQUIRED.issubset(data.columns):
        for sid in data.get(
            "sentence_id",
            pd.Series(dtype=int),
        ).tolist():
            repairs.append({
                "pair": pair_name,
                "mode": mode,
                "sentence_id": sid,
                "reason": "incompatible_or_legacy_schema",
                "checkpoint": path,
            })

        _record_repairs(repairs)
        _atomic_to_csv(pd.DataFrame(), path)
        return pd.DataFrame()

    if mode == "free_generation" and not {
        "hyp_text",
        "generated_steps",
        "generation_hit_max_length",
        "attention_inference_eligible",
        "attention_tensor_role",
    }.issubset(data.columns):
        for sid in data.sentence_id.tolist():
            repairs.append({
                "pair": pair_name,
                "mode": mode,
                "sentence_id": sid,
                "reason": "missing_free_generation_columns",
                "checkpoint": path,
            })

        _record_repairs(repairs)
        _atomic_to_csv(pd.DataFrame(), path)
        return pd.DataFrame()

    data["sentence_id"] = pd.to_numeric(
        data["sentence_id"],
        errors="coerce",
    )

    valid_rows = []
    seen = set()
    expected = set(int(x) for x in expected_ids)

    for index in reversed(data.index.tolist()):
        row = data.loc[index]
        sid_value = row["sentence_id"]
        reason = None

        if pd.isna(sid_value):
            reason = "non_numeric_sentence_id"
            sid = -1
        else:
            sid = int(sid_value)

            if sid not in expected:
                reason = "unexpected_sentence_id"

            elif sid in seen:
                reason = "duplicate_sentence_id"

            elif (
                str(row["pipeline_schema_version"])
                != PIPELINE_SCHEMA_VERSION
            ):
                reason = "schema_version_mismatch"

            elif int(row["source_tags_removed"]) != 1:
                reason = "source_tag_audit_failed"

            elif int(row["source_special_tokens_removed"]) != 2:
                reason = "source_control_audit_failed"

            elif str(row["target_row_alignment"]) != (
                "predicted_generated_token"
                if mode == "free_generation"
                else "predicted_label_token"
            ):
                reason = "target_row_alignment_mismatch"

            elif require_tensor:
                is_cap = (
                    mode == "free_generation"
                    and int(row.get("generation_hit_max_length", 0)) == 1
                )
                expected_tensor = (
                    os.path.join(ROOT_DIR, "free_generation_capped", pair_name, f"{sid}.npy")
                    if is_cap else os.path.join(tensor_dir, f"{sid}.npy")
                )
                if not _tensor_is_valid(expected_tensor):
                    reason = "missing_or_invalid_diagnostic_tensor" if is_cap else "missing_or_invalid_tensor"

        if reason is None:
            valid_rows.append(index)
            seen.add(sid)
        else:
            repairs.append({
                "pair": pair_name,
                "mode": mode,
                "sentence_id": sid,
                "reason": reason,
                "checkpoint": path,
            })

    clean = data.loc[sorted(valid_rows)].copy()

    if not clean.empty:
        clean["sentence_id"] = clean["sentence_id"].astype(int)
        clean = clean.sort_values("sentence_id")

    if repairs:
        _record_repairs(repairs)
        _atomic_to_csv(clean, path)

        print(
            f"Resume repaired {len(repairs)} invalid row(s): "
            f"{pair_name}/{mode}",
            flush=True,
        )

    return clean


def _count(data):
    if "sentence_id" not in data.columns:
        return 0

    return int(data.sentence_id.nunique())


FAILURE_COLUMNS = [
    "pair",
    "sentence_id",
    "mode",
    "error",
]

if (
    RESUME
    and os.path.isfile(log_path)
    and os.path.getsize(log_path) > 0
):
    try:
        failures = pd.read_csv(log_path).to_dict("records")
    except pd.errors.EmptyDataError:
        failures = []
else:
    failures = []


def _write_failures():
    frame = (
        pd.DataFrame(failures, columns=FAILURE_COLUMNS)
        .drop_duplicates()
    )
    _atomic_to_csv(frame, log_path)
    return frame


def sync_relative_paths(relative_paths):
    if not AUTO_DRIVE_BACKUP:
        return

    assert DRIVE_DESTINATION is not None
    assert Path("/content/drive/MyDrive").is_dir(), (
        "Google Drive is not mounted"
    )

    for relative_path in sorted(set(relative_paths)):
        source_path = Path(ROOT_DIR, relative_path)

        if not source_path.exists():
            continue

        destination = Path(
            DRIVE_DESTINATION,
            relative_path,
        )

        if source_path.is_dir():
            destination.mkdir(
                parents=True,
                exist_ok=True,
            )
            command = [
                "rsync",
                "-r",
                "--checksum",
                "--partial",
                str(source_path) + "/",
                str(destination) + "/",
            ]
        else:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            command = [
                "rsync",
                "--checksum",
                "--partial",
                str(source_path),
                str(destination),
            ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )


def sync_control_files():
    relative_paths = []

    for folder in ("manifests", "logs"):
        directory = Path(ROOT_DIR, folder)

        if directory.is_dir():
            relative_paths.extend(
                str(item.relative_to(ROOT_DIR))
                for item in directory.iterdir()
                if item.is_file()
            )

    sync_relative_paths(relative_paths)


def sync_pair_checkpoints(pair_names):
    paths = []

    for pair_name in pair_names:
        paths.extend([
            f"teacher_forced/{pair_name}",
            f"teacher_forced/{pair_name}_stats.csv",
            f"free_generation/{pair_name}",
            f"free_generation/{pair_name}_stats.csv",
            f"free_generation_capped/{pair_name}",
        ])

    sync_relative_paths(paths)
    sync_control_files()

    print(
        f"Drive checkpoint synced for "
        f"{len(pair_names)} pair(s): {pair_names}",
        flush=True,
    )


pending_drive_pairs = []
all_tf = []
all_free = []

for p_idx, prow in pair_df.iterrows():
    seed_everything(SEED + p_idx)

    pair_name = prow.pair
    pair_data = get_pair_data(prow)

    tf_csv = os.path.join(
        tf_dir,
        f"{pair_name}_stats.csv",
    )
    free_csv = os.path.join(
        free_dir,
        f"{pair_name}_stats.csv",
    )

    tf_tensor_dir = os.path.join(
        tf_dir,
        pair_name,
    )
    free_tensor_dir = os.path.join(
        free_dir,
        pair_name,
    )
    free_capped_tensor_dir = os.path.join(
        free_capped_dir,
        pair_name,
    )

    os.makedirs(tf_tensor_dir, exist_ok=True)
    os.makedirs(free_tensor_dir, exist_ok=True)
    os.makedirs(free_capped_tensor_dir, exist_ok=True)

    tf_pair = _load_checkpoint(
        tf_csv,
        tf_tensor_dir,
        pair_name,
        "teacher_forced",
        range(N_SENTENCES),
        True,
    )

    free_pair = _load_checkpoint(
        free_csv,
        free_tensor_dir,
        pair_name,
        "free_generation",
        FREE_SENTENCE_IDS,
        SAVE_FULL_TENSORS_FREE,
    )

    tf_done = set(
        tf_pair.get(
            "sentence_id",
            pd.Series(dtype=int),
        ).astype(int)
    )

    free_done = set(
        free_pair.get(
            "sentence_id",
            pd.Series(dtype=int),
        ).astype(int)
    )

    print(
        f"\n[{p_idx + 1}/{len(pair_df)}] {pair_name}: "
        f"TF={len(tf_done)}/{N_SENTENCES}, "
        f"FREE={len(free_done)}/{N_FREE}",
        flush=True,
    )

    # --------------------------------------------------------
    # Teacher-forced extraction
    # --------------------------------------------------------
    if RUN_TEACHER_FORCED:
        missing = [
            sid
            for sid in range(N_SENTENCES)
            if sid not in tf_done
        ]

        for batch_idx, start in enumerate(
            range(0, len(missing), BATCH_SIZE_TF),
            1,
        ):
            sids = missing[
                start:start + BATCH_SIZE_TF
            ]

            srcs = pair_data.iloc[sids].source.tolist()
            tgts = pair_data.iloc[sids].target.tolist()

            show = (
                batch_idx == 1
                or batch_idx % PRINT_EVERY_TF_BATCHES == 0
                or start + len(sids) == len(missing)
            )

            if show:
                print(
                    f"[{p_idx + 1}/40] {pair_name} "
                    f"TF {sids[0]}..{sids[-1]} START",
                    flush=True,
                )

            try:
                results = (
                    extract_attention_teacher_forced_batch(
                        prow.src_lang,
                        prow.tgt_lang,
                        srcs,
                        tgts,
                    )
                )

            except Exception as exc:
                print(
                    f"TF batch failed: {exc}; "
                    "retrying individual fixed IDs",
                    flush=True,
                )

                results = []

                for sid, src, tgt in zip(
                    sids,
                    srcs,
                    tgts,
                ):
                    try:
                        result = (
                            extract_attention_teacher_forced_batch(
                                prow.src_lang,
                                prow.tgt_lang,
                                [src],
                                [tgt],
                            )[0]
                        )
                        results.append(result)

                    except Exception as item_exc:
                        failures.append({
                            "pair": pair_name,
                            "sentence_id": sid,
                            "mode": "teacher_forced",
                            "error": str(item_exc),
                        })
                        results.append(None)

            for sid, result in zip(sids, results):
                if result is None:
                    failures.append({
                        "pair": pair_name,
                        "sentence_id": sid,
                        "mode": "teacher_forced_empty",
                        "error": "No retained tokens",
                    })
                    continue

                full, stat = result

                assert np.isfinite(full).all()

                _atomic_save_npy(
                    full.astype(np.float16),
                    os.path.join(
                        tf_tensor_dir,
                        f"{sid}.npy",
                    ),
                )

                stat.update({
                    "pair": pair_name,
                    "relation": prow.relation,
                    "family_relation": prow.family_relation,
                    "sentence_id": sid,
                    "corpus_sentence_id": str(
                        pair_data.iloc[sid].sentence_id
                    ),
                })

                tf_pair = pd.concat(
                    [
                        tf_pair,
                        pd.DataFrame([stat]),
                    ],
                    ignore_index=True,
                )

            _write_checkpoint(tf_pair, tf_csv)
            _write_failures()

            if show:
                print(
                    f"[{p_idx + 1}/40] {pair_name} "
                    f"TF {_count(tf_pair)}/{N_SENTENCES} DONE",
                    flush=True,
                )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Free-generation extraction
    # --------------------------------------------------------
    if RUN_FREE_GENERATION:
        seed_everything(SEED + p_idx)

        missing = [
            sid
            for sid in FREE_SENTENCE_IDS
            if sid not in free_done
        ]

        for batch_idx, start in enumerate(
            range(0, len(missing), BATCH_SIZE_FREE),
            1,
        ):
            sids = missing[
                start:start + BATCH_SIZE_FREE
            ]

            srcs = pair_data.iloc[sids].source.tolist()

            try:
                results = extract_attention_free_batch(
                    prow.src_lang, prow.tgt_lang, srcs,
                    max_length=FREE_MAX_LENGTH,
                )

            except Exception as exc:
                for sid in sids:
                    failures.append({
                        "pair": pair_name,
                        "sentence_id": sid,
                        "mode": "free_generation",
                        "error": str(exc),
                    })

                results = [None] * len(sids)

            for sid, src, result in zip(
                sids,
                srcs,
                results,
            ):
                if result is None:
                    try:
                        result = extract_attention_free_batch(
                            prow.src_lang, prow.tgt_lang, [src],
                            max_length=FREE_MAX_LENGTH,
                        )[0]

                    except Exception as exc:
                        failures.append({
                            "pair": pair_name,
                            "sentence_id": sid,
                            "mode": "free_generation",
                            "error": str(exc),
                        })

                if result is None:
                    failures.append({
                        "pair": pair_name,
                        "sentence_id": sid,
                        "mode": "free_generation_empty",
                        "error": "No retained tokens",
                    })
                    continue

                full, stat, hypothesis = result

                assert np.isfinite(full).all()

                hit_cap = int(stat["generation_hit_max_length"])

                stat.update({
                    "pair": pair_name,
                    "relation": prow.relation,
                    "family_relation": prow.family_relation,
                    "sentence_id": sid,
                    "corpus_sentence_id": str(pair_data.iloc[sid].sentence_id),
                    "hyp_text": hypothesis,
                    "attention_inference_eligible": int(hit_cap == 0),
                    "attention_tensor_role": (
                        "inferential" if hit_cap == 0
                        else "diagnostic_capped_excluded"
                    ),
                })

                if hit_cap:
                    print(
                        f"WARNING: {pair_name} sentence {sid} reached total "
                        f"FREE_MAX_LENGTH={FREE_MAX_LENGTH}; hypothesis retained for "
                        "quality, diagnostic tensor separated, attention excluded.",
                        flush=True,
                    )
                    tensor_destination = os.path.join(
                        free_capped_tensor_dir, f"{sid}.npy"
                    )
                    for key in list(stat):
                        if (
                            key.startswith("diagonal_")
                            or key.startswith("uniform_diagonal_")
                            or key.startswith("all_retained_diagonal_")
                            or key.startswith("all_retained_uniform_diagonal_")
                            or key.startswith("layer_diagonal_")
                            or key.startswith("layer_expected_")
                            or key in {
                                "mean", "std", "min", "max",
                                "expected_normalized_displacement", "normalized_entropy",
                                "mean_row_max", "row_max_excess", "diagonal_excess",
                                "all_retained_mean", "all_retained_std",
                                "all_retained_min", "all_retained_max",
                                "all_retained_expected_normalized_displacement",
                                "all_retained_normalized_entropy",
                                "all_retained_mean_row_max", "all_retained_row_max_excess",
                                "all_retained_diagonal_excess",
                            }
                        ):
                            stat[key] = "[]" if key.endswith("_json") else np.nan
                    stat["analysis_view"] = "excluded_generation_cap"
                else:
                    tensor_destination = os.path.join(free_tensor_dir, f"{sid}.npy")

                if SAVE_FULL_TENSORS_FREE:
                    _atomic_save_npy(full.astype(np.float16), tensor_destination)

                free_pair = pd.concat(
                    [
                        free_pair,
                        pd.DataFrame([stat]),
                    ],
                    ignore_index=True,
                )

            _write_checkpoint(
                free_pair,
                free_csv,
            )
            _write_failures()

            if (
                batch_idx == 1
                or batch_idx % PRINT_EVERY_FREE_ITEMS == 0
                or start + len(sids) == len(missing)
            ):
                print(
                    f"[{p_idx + 1}/40] {pair_name} "
                    f"FREE: {_count(free_pair)}/{N_FREE}",
                    flush=True,
                )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Reconcile before declaring the pair complete.
    tf_pair = _load_checkpoint(
        tf_csv,
        tf_tensor_dir,
        pair_name,
        "teacher_forced",
        range(N_SENTENCES),
        True,
    )

    free_pair = _load_checkpoint(
        free_csv,
        free_tensor_dir,
        pair_name,
        "free_generation",
        FREE_SENTENCE_IDS,
        SAVE_FULL_TENSORS_FREE,
    )

    all_tf.append(tf_pair)
    all_free.append(free_pair)

    _write_failures()
    pending_drive_pairs.append(pair_name)

    if (
        AUTO_DRIVE_BACKUP
        and (
            len(pending_drive_pairs)
            >= DRIVE_SYNC_EVERY_PAIRS
            or p_idx == len(pair_df) - 1
        )
    ):
        sync_pair_checkpoints(
            pending_drive_pairs
        )
        pending_drive_pairs.clear()


TF = pd.concat(
    all_tf,
    ignore_index=True,
)

FREE = pd.concat(
    all_free,
    ignore_index=True,
)

# Reconstruct eligibility for resumed rows that were written before the
# eligibility column was introduced.
FREE["generation_hit_max_length"] = (
    FREE["generation_hit_max_length"]
    .fillna(0)
    .astype(int)
)

FREE["attention_inference_eligible"] = (
    FREE["generation_hit_max_length"] == 0
).astype(int)

FREE["attention_tensor_role"] = np.where(
    FREE["attention_inference_eligible"] == 1,
    "inferential",
    "diagnostic_capped_excluded",
)

# Remove capped attention summaries from the inferential global table.
# Token IDs, hypothesis text, lengths and diagnostic fields remain intact.
attention_columns_to_exclude = [
    column
    for column in FREE.columns
    if (
        column.startswith("diagonal_")
        or column.startswith("uniform_diagonal_")
        or column.startswith("all_retained_diagonal_")
        or column.startswith("all_retained_uniform_diagonal_")
        or column.startswith("layer_diagonal_")
        or column.startswith("layer_expected_")
        or column in {
            "expected_normalized_displacement",
            "normalized_entropy",
            "mean_row_max",
            "row_max_excess",
            "all_retained_expected_normalized_displacement",
            "all_retained_normalized_entropy",
            "all_retained_mean_row_max",
            "all_retained_row_max_excess",
            "mean", "std", "min", "max",
            "all_retained_mean", "all_retained_std",
            "all_retained_min", "all_retained_max",
        }
    )
]

capped_mask = (
    FREE["generation_hit_max_length"] == 1
)

FREE.loc[
    capped_mask,
    attention_columns_to_exclude,
] = np.nan

tf_all_path = os.path.join(
    ROOT_DIR,
    "teacher_forced_all_stats.csv",
)

free_all_path = os.path.join(
    ROOT_DIR,
    "free_generation_all_stats.csv",
)

_atomic_to_csv(TF, tf_all_path)
_atomic_to_csv(FREE, free_all_path)

failure_df = _write_failures()
cap_audit = FREE.loc[FREE["generation_hit_max_length"].eq(1), [
    "pair", "sentence_id", "corpus_sentence_id", "generated_steps",
    "generation_ended_with_eos", "target_special_tokens_removed",
    "attention_inference_eligible", "attention_tensor_role", "hyp_text",
]].copy()
_atomic_to_csv(cap_audit, cap_audit_path)


def reconcile_failures(
    failure_frame,
    tf_frame,
    free_frame,
):
    done_tf = set(
        zip(
            tf_frame.get("pair", []),
            tf_frame.get("sentence_id", []),
        )
    )

    done_free = set(
        zip(
            free_frame.get("pair", []),
            free_frame.get("sentence_id", []),
        )
    )

    audit = failure_frame.copy()
    states = []

    for row in audit.itertuples(index=False):
        completed = (
            done_tf
            if str(row.mode).startswith("teacher_forced")
            else done_free
        )

        states.append(
            "resolved"
            if (
                row.pair,
                int(row.sentence_id),
            ) in completed
            else "unresolved"
        )

    audit["status"] = states
    return audit


failure_audit = reconcile_failures(
    failure_df,
    TF,
    FREE,
)

failure_audit_path = os.path.join(
    ROOT_DIR,
    "logs",
    "failure_audit.csv",
)

_atomic_to_csv(
    failure_audit,
    failure_audit_path,
)

unresolved = failure_audit[
    failure_audit["status"] == "unresolved"
]

assert unresolved.empty, (
    "Unresolved fixed IDs; resume: "
    f"{unresolved.head(10).to_dict('records')}"
)

validate_collected_ids(
    TF,
    range(N_SENTENCES),
    "TF",
)

validate_collected_ids(
    FREE,
    FREE_SENTENCE_IDS,
    "FREE",
)

assert len(TF) == len(pair_df) * N_SENTENCES
assert len(FREE) == len(pair_df) * N_FREE

assert (
    TF["pipeline_schema_version"]
    == PIPELINE_SCHEMA_VERSION
).all()

assert (
    FREE["pipeline_schema_version"]
    == PIPELINE_SCHEMA_VERSION
).all()

assert (
    TF["source_tags_removed"].astype(int) == 1
).all()

assert (
    TF["source_special_tokens_removed"].astype(int) == 2
).all()

assert (
    FREE["source_tags_removed"].astype(int) == 1
).all()

assert (
    FREE["source_special_tokens_removed"].astype(int) == 2
).all()

free_cap_hits = int(
    FREE["generation_hit_max_length"].sum()
)

free_attention_eligible = int(
    FREE["attention_inference_eligible"].sum()
)

print(
    f"Free-generation cap hits: "
    f"{free_cap_hits}/{len(FREE)}"
)

print(
    f"Attention-inference eligible free rows: "
    f"{free_attention_eligible}/{len(FREE)}"
)

if free_cap_hits:
    print(
        "Capped hypotheses are retained as translation failures, "
        "but their attention summaries are excluded."
    )

tf_counts = (
    TF.groupby("pair")
    .sentence_id.nunique()
    .reindex(pair_df.pair, fill_value=0)
)

free_counts = (
    FREE.groupby("pair")
    .sentence_id.nunique()
    .reindex(pair_df.pair, fill_value=0)
)

eligible_counts = (
    FREE[
        FREE["attention_inference_eligible"] == 1
    ]
    .groupby("pair")
    .sentence_id.nunique()
    .reindex(pair_df.pair, fill_value=0)
)

collection_verification = {
    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
    "teacher_forced_rows": len(TF),
    "free_generation_rows": len(FREE),
    "pairs": len(pair_df),
    "teacher_forced_per_pair_min": int(
        tf_counts.min()
    ),
    "teacher_forced_per_pair_max": int(
        tf_counts.max()
    ),
    "free_generation_per_pair_min": int(
        free_counts.min()
    ),
    "free_generation_per_pair_max": int(
        free_counts.max()
    ),
    "free_attention_eligible_rows": (
        free_attention_eligible
    ),
    "free_attention_eligible_per_pair_min": int(
        eligible_counts.min()
    ),
    "free_attention_eligible_per_pair_max": int(
        eligible_counts.max()
    ),
    "checkpoint_rows_pruned": len(
        checkpoint_repairs
    ),
    "logged_failures": len(
        failure_df
    ),
    "unresolved_failures": len(
        unresolved
    ),
    "free_generation_length_boundary_hits": (
        free_cap_hits
    ),
    "cap_policy": (
        "retain hypothesis for quality; store tensor only under "
        "free_generation_capped; exclude attention from inference"
    ),
}

with open(
    os.path.join(
        ROOT_DIR,
        "manifests",
        "collection_verification.json",
    ),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        collection_verification,
        handle,
        indent=2,
    )

COLLECTION_VERIFIED = True

print(
    "COLLECTION VERIFIED:",
    collection_verification,
    flush=True,
)

sync_relative_paths([
    "teacher_forced_all_stats.csv",
    "free_generation_all_stats.csv",
    "manifests/collection_verification.json",
    "manifests/generation_cap_audit.csv",
    "logs/failure_audit.csv",
])

sync_control_files()

# %% original code cell 18; notebook cell 25
# ============================================================
# 13. FLORES-200 metrics and DATA-safe matched references
# ============================================================
def metric_texts(texts):
    values = ["" if x is None else str(x) for x in texts]
    assert all("\n" not in x and "\r" not in x for x in values), (
        "Metric inputs must remain one physical line; refusing to mutate text."
    )
    return values

# Construction downloads/caches the canonical FLORES-200 scoring SPM once.
SPBLEU_METRIC = sacrebleu.metrics.BLEU(tokenize="flores200")
CHRF_METRIC = sacrebleu.metrics.CHRF(char_order=6, word_order=0, beta=2)
CHRFPP_METRIC = sacrebleu.metrics.CHRF(char_order=6, word_order=2, beta=2)
SPBLEU_TOKENIZER_SHA256 = hashlib.sha256(
    SPBLEU_METRIC.tokenizer.sp.serialized_model_proto()
).hexdigest()

def score_flores(hypotheses, references):
    hypotheses, references = metric_texts(hypotheses), metric_texts(references)
    assert len(hypotheses) == len(references) > 0
    bleu = SPBLEU_METRIC.corpus_score(hypotheses, [references]).score
    chrf = CHRF_METRIC.corpus_score(hypotheses, [references]).score
    chrfpp = CHRFPP_METRIC.corpus_score(hypotheses, [references]).score
    return bleu, chrf, chrfpp

_ = score_flores(["metric probe"], ["metric probe"])
METRIC_SIGNATURES = {
    "spbleu": str(SPBLEU_METRIC.get_signature()),
    "chrf": str(CHRF_METRIC.get_signature()),
    "chrfpp": str(CHRFPP_METRIC.get_signature()),
    "flores200_tokenizer_sha256": SPBLEU_TOKENIZER_SHA256,
}
with open(os.path.join(ROOT_DIR, "manifests", "metric_signatures.json"), "w") as handle:
    json.dump(METRIC_SIGNATURES, handle, indent=2)
print("Metric signatures:", METRIC_SIGNATURES)

reference_parts = []
for _, prow in pair_df.iterrows():
    part = get_pair_data(prow)[["run_sentence_id", "sentence_id", "target"]].copy()
    part = part.rename(columns={
        "run_sentence_id": "sentence_id",
        "sentence_id": "corpus_sentence_id_reference",
        "target": "reference",
    })
    part["pair"] = prow.pair
    part["tgt_lang"] = prow.tgt_lang
    reference_parts.append(part)
REFERENCE_TABLE = pd.concat(reference_parts, ignore_index=True)
assert not REFERENCE_TABLE.duplicated(["pair", "sentence_id"]).any()
FREE_MATCHED = FREE.merge(
    REFERENCE_TABLE, on=["pair", "sentence_id"], how="left", validate="one_to_one"
)
assert FREE_MATCHED["reference"].notna().all()
assert (
    FREE_MATCHED["corpus_sentence_id"].astype(str)
    == FREE_MATCHED["corpus_sentence_id_reference"].astype(str)
).all(), "Stored and reconstructed corpus sentence IDs differ."


# %% original code cell 19; notebook cell 26
# ============================================================
# 14. Free-generation diagnostics (text, subword, repetition, cap proximity)
# ============================================================
def _max_ngram_share(tokens, n):
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))]
    if not grams:
        return 0.0
    counts = pd.Series(grams).value_counts()
    # A high share caused by one unique n-gram in a short sentence is not repetition.
    if int(counts.iloc[0]) < 2:
        return 0.0
    return float(counts.iloc[0] / len(grams))

def _character_run_diagnostics(text):
    chars = [ch for ch in str(text) if not ch.isspace()]
    if not chars:
        return 0, 0.0
    longest = current = 1
    for left, right in zip(chars, chars[1:]):
        current = current + 1 if right == left else 1
        longest = max(longest, current)
    return int(longest), float(longest / len(chars))

def hypothesis_diagnostics(hypothesis, reference, lang_code,
                           generated_subwords, reference_subwords, generated_steps):
    hypothesis = "" if hypothesis is None else str(hypothesis)
    reference = "" if reference is None else str(reference)
    script_markers = {
        "Deva": "DEVANAGARI", "Beng": "BENGALI", "Guru": "GURMUKHI",
        "Gujr": "GUJARATI", "Orya": "ORIYA", "Knda": "KANNADA",
        "Mlym": "MALAYALAM", "Taml": "TAMIL", "Telu": "TELUGU",
        "Arab": "ARABIC", "Latn": "LATIN",
    }
    scripts = set()
    for character in hypothesis:
        name = unicodedata.name(character, "")
        for script, marker in script_markers.items():
            if marker in name or (script == "Orya" and "ODIA" in name):
                scripts.add(script)
    hyp_tokens, ref_tokens = hypothesis.split(), reference.split()
    hyp_len, ref_len = len(hyp_tokens), len(ref_tokens)
    expected_script = lang_code.split("_")[-1]
    placeholder_pattern = re.compile(
        r"(?:<|\[)\s*(?:ID|id|आई\s*\.?\s*डी|आइ\s*\.?\s*डी)\s*\d+\s*(?:>|\])"
    )
    unique_ratio = len(set(hyp_tokens)) / hyp_len if hyp_len else 0.0
    longest_char_run, longest_char_run_ratio = _character_run_diagnostics(hypothesis)
    return {
        "hyp_empty": int(not hypothesis.strip()),
        "replacement_count": hypothesis.count("\ufffd"),
        "control_count": sum(1 for ch in hypothesis
                             if unicodedata.category(ch).startswith("C") and ch not in "\n\t"),
        "placeholder_pattern_count": len(placeholder_pattern.findall(hypothesis)),
        "scripts": ",".join(sorted(scripts)),
        "expected_script": expected_script,
        "expected_script_absent": int(bool(scripts) and expected_script not in scripts),
        "hyp_len": hyp_len, "ref_len": ref_len,
        "length_ratio": hyp_len / ref_len if ref_len else np.nan,
        "unique_token_ratio": unique_ratio,
        "repetition_ratio": 1.0 - unique_ratio if hyp_len else 0.0,
        "max_bigram_share": _max_ngram_share(hyp_tokens, 2),
        "max_trigram_share": _max_ngram_share(hyp_tokens, 3),
        "longest_character_run": longest_char_run,
        "longest_character_run_ratio": longest_char_run_ratio,
        "generated_subword_count": int(generated_subwords),
        "reference_subword_count": int(reference_subwords),
        "subword_length_ratio": (generated_subwords / reference_subwords
                                 if reference_subwords else np.nan),
        "generated_steps": int(generated_steps),
        "near_generation_cap": int((1 + generated_steps) >= 0.90 * FREE_MAX_LENGTH),
    }

# Reference subword lengths use the same NLLB target mode as TF labels.
reference_subword_counts = {}
for prow in pair_df.itertuples(index=False):
    group = REFERENCE_TABLE[REFERENCE_TABLE.pair == prow.pair].sort_values("sentence_id")
    configure_languages(prow.src_lang, prow.tgt_lang)
    encoded = tokenizer(
        text_target=group.reference.astype(str).tolist(),
        add_special_tokens=True, truncation=False,
    )
    for sid, ids in zip(group.sentence_id.astype(int), encoded["input_ids"]):
        assert ids[0] == language_id(prow.tgt_lang) and ids[-1] == tokenizer.eos_token_id
        reference_subword_counts[(prow.pair, sid)] = len(ids) - 2

diagnostic_rows = []
for row in FREE_MATCHED.itertuples(index=False):
    diag = hypothesis_diagnostics(
        row.hyp_text, row.reference, row.tgt_lang,
        generated_subwords=int(row.tgt_retained_tokens),
        reference_subwords=reference_subword_counts[(row.pair, int(row.sentence_id))],
        generated_steps=int(row.generated_steps),
    )
    generated_pieces = json.loads(row.target_tokens_json)
    diag.update({
        "pair": row.pair, "sentence_id": int(row.sentence_id),
        "generated_unknown_tokens": int(row.target_unknown_tokens),
        "generation_hit_max_length": int(row.generation_hit_max_length),
        "subword_max_bigram_share": _max_ngram_share(generated_pieces, 2),
        "subword_max_trigram_share": _max_ngram_share(generated_pieces, 3),
        "lexical_view_unavailable": int(row.analysis_view != "lexical_no_punctuation"),
    })
    diagnostic_rows.append(diag)
free_quality_diag = pd.DataFrame(diagnostic_rows)
free_quality_diag["quality_flag"] = (
    (free_quality_diag.hyp_empty == 1)
    | (free_quality_diag.replacement_count > 0)
    | (free_quality_diag.control_count > 0)
    | (free_quality_diag.placeholder_pattern_count > 0)
    | (free_quality_diag.expected_script_absent == 1)
    | (free_quality_diag.length_ratio < 0.25)
    | (free_quality_diag.length_ratio > 4.0)
    | (free_quality_diag.subword_length_ratio < 0.25)
    | (free_quality_diag.subword_length_ratio > 4.0)
    | (free_quality_diag.unique_token_ratio < 0.25)
    | (free_quality_diag.max_bigram_share > 0.50)
    | (free_quality_diag.max_trigram_share > 0.40)
    | (free_quality_diag.subword_max_bigram_share > 0.50)
    | (free_quality_diag.subword_max_trigram_share > 0.40)
    | ((free_quality_diag.longest_character_run >= 8)
       & (free_quality_diag.longest_character_run_ratio > 0.20))
    | (free_quality_diag.generated_unknown_tokens > 0)
    | (free_quality_diag.generation_hit_max_length == 1)
    | (free_quality_diag.near_generation_cap == 1)
    | (free_quality_diag.lexical_view_unavailable == 1)
)
diagnostics_path = os.path.join(ROOT_DIR, "manifests", "free_hypothesis_diagnostics.csv")
_atomic_to_csv(free_quality_diag, diagnostics_path)
print("Flagged hypotheses:", int(free_quality_diag.quality_flag.sum()), "/", len(free_quality_diag))
print(free_quality_diag.groupby("pair").quality_flag.mean().sort_values(ascending=False).head(20))


# %% original code cell 20; notebook cell 27
diagnostic_columns = [
    "pair", "sentence_id", "hyp_empty", "replacement_count", "control_count",
    "expected_script_absent", "placeholder_pattern_count", "hyp_len", "ref_len",
    "length_ratio", "generated_subword_count", "reference_subword_count",
    "subword_length_ratio", "unique_token_ratio", "max_bigram_share",
    "max_trigram_share", "subword_max_bigram_share", "subword_max_trigram_share",
    "longest_character_run", "longest_character_run_ratio",
    "generated_unknown_tokens", "generation_hit_max_length",
    "near_generation_cap", "lexical_view_unavailable",
    "quality_flag",
]
display(free_quality_diag.loc[free_quality_diag.quality_flag,
       [c for c in diagnostic_columns if c in free_quality_diag.columns]]
       .sort_values(["pair", "sentence_id"]))


# %% original code cell 21; notebook cell 28
# ============================================================
# 15. Matched greedy spBLEU, chrF, chrF++ and sentence chrF++
# ============================================================
sentence_quality_parts, matched_rows = [], []
for p_idx, prow in pair_df.iterrows():
    group = FREE_MATCHED[FREE_MATCHED.pair == prow.pair].sort_values("sentence_id").copy()
    assert len(group) == FREE_N_SENTENCES
    hypotheses = metric_texts(group.hyp_text.tolist())
    references = metric_texts(group.reference.tolist())
    group["sentence_chrfpp"] = [
        CHRFPP_METRIC.sentence_score(h, [r]).score
        for h, r in zip(hypotheses, references)
    ]
    sentence_quality_parts.append(group[["pair", "sentence_id", "corpus_sentence_id",
                                         "sentence_chrfpp"]])
    spbleu, chrf, chrfpp = score_flores(hypotheses, references)
    matched_rows.append({
        "pair": prow.pair, "relation": prow.relation,
        "family_relation": prow.family_relation, "src_lang": prow.src_lang,
        "tgt_lang": prow.tgt_lang,
        # Compatibility alias: this is specifically FLORES-200 spBLEU.
        "bleu_greedy": spbleu, "spbleu_greedy": spbleu,
        "chrf_greedy": chrf, "chrfpp_greedy": chrfpp,
        "mean_sentence_chrfpp": float(group.sentence_chrfpp.mean()),
        "median_sentence_chrfpp": float(group.sentence_chrfpp.median()),
        "n_scored": len(group),
    })
    print(f"[{p_idx+1}/40] {prow.pair}: spBLEU={spbleu:.2f} "
          f"chrF={chrf:.2f} chrF++={chrfpp:.2f}")

sentence_quality = pd.concat(sentence_quality_parts, ignore_index=True)
_atomic_to_csv(sentence_quality, os.path.join(ROOT_DIR, "manifests", "sentence_quality_greedy.csv"))
quality_free_matched = pd.DataFrame(matched_rows)
assert len(quality_free_matched) == 40 and quality_free_matched.pair.is_unique
assert set(quality_free_matched.pair) == set(pair_df.pair)
assert (quality_free_matched.n_scored == FREE_N_SENTENCES).all()
_atomic_to_csv(quality_free_matched,
               os.path.join(ROOT_DIR, "manifests", "quality_free_matched.csv"))

# Cross-model quality is diagnostic only. Re-score the completed IndicTrans2
# hypotheses with this exact same raw-text FLORES metric pipeline; never compare
# its native preprocessed BLEU/chrF numbers directly with NLLB scores.
_indic_root = Path(INDICTRANS_HOLDOUT_DRIVE_ROOT)
_indic_free_path = _indic_root / "free_generation_all_stats.csv"
assert _indic_free_path.is_file(), (
    "Completed IndicTrans2 holdout is required for common cross-model quality: "
    f"{_indic_free_path}"
)
_indic_free = pd.read_csv(_indic_free_path, dtype={"corpus_sentence_id": str})
_indic_free = _indic_free[_indic_free.sentence_id.astype(int).isin(FREE_SENTENCE_IDS)].copy()
assert len(_indic_free) == 40 * FREE_N_SENTENCES
_indic_matched = _indic_free.merge(
    REFERENCE_TABLE,
    on=["pair", "sentence_id"], how="left", validate="one_to_one",
)
assert _indic_matched.reference.notna().all()
assert (
    _indic_matched.corpus_sentence_id.astype(str)
    == _indic_matched.corpus_sentence_id_reference.astype(str)
).all(), "IndicTrans2 and NLLB common-score corpus identities differ."

_common_quality_rows = []
for prow in pair_df.itertuples(index=False):
    for model_label, frame in (
        ("IndicTrans2-320M", _indic_matched),
        ("NLLB-200-distilled-600M", FREE_MATCHED),
    ):
        group = frame[frame.pair == prow.pair].sort_values("sentence_id")
        assert len(group) == FREE_N_SENTENCES
        spbleu, chrf, chrfpp = score_flores(
            group.hyp_text.tolist(), group.reference.tolist()
        )
        _common_quality_rows.append({
            "model": model_label, "pair": prow.pair,
            "relation": prow.relation, "family_relation": prow.family_relation,
            "src_lang": prow.src_lang, "tgt_lang": prow.tgt_lang,
            "n_scored": len(group), "common_spbleu": spbleu,
            "common_chrf": chrf, "common_chrfpp": chrfpp,
            "metric_signature_sha256": hashlib.sha256(
                json.dumps(METRIC_SIGNATURES, sort_keys=True).encode()
            ).hexdigest(),
        })
common_cross_model_quality = pd.DataFrame(_common_quality_rows)
assert len(common_cross_model_quality) == 80
assert common_cross_model_quality.groupby("model").pair.nunique().eq(40).all()
_atomic_to_csv(
    common_cross_model_quality,
    os.path.join(ROOT_DIR, "manifests", "common_cross_model_quality.csv"),
)
print("Saved common raw-text FLORES quality for both models:")
display(common_cross_model_quality.groupby("model")[[
    "common_spbleu", "common_chrf", "common_chrfpp"
]].mean())

attention_columns = [
    "diagonal_mass_5pct", "diagonal_excess_5pct",
    "diagonal_mass_10pct", "diagonal_excess_10pct",
    "diagonal_mass_20pct", "diagonal_excess_20pct",
    "expected_normalized_displacement", "normalized_entropy",
    "mean_row_max", "row_max_excess", "src_len", "tgt_len",
]
validate_collected_ids(FREE_MATCHED, FREE_SENTENCE_IDS, "FREE_MATCHED")
FREE_ATTENTION_ELIGIBLE = FREE_MATCHED.loc[
    FREE_MATCHED.attention_inference_eligible.astype(int).eq(1)
].copy()
assert not FREE_ATTENTION_ELIGIBLE.generation_hit_max_length.astype(int).any()
matched_attention = FREE_ATTENTION_ELIGIBLE.groupby("pair")[attention_columns].mean().add_prefix(
    "free_mean_"
).reset_index()
total_counts = FREE_MATCHED.groupby("pair").size()
eligible_counts = FREE_ATTENTION_ELIGIBLE.groupby("pair").size()
lexical_counts = FREE_ATTENTION_ELIGIBLE.groupby("pair").diagonal_excess_10pct.count()
matched_attention["n_generation_total"] = matched_attention.pair.map(total_counts)
matched_attention["n_attention_eligible"] = matched_attention.pair.map(eligible_counts)
matched_attention["n_attention_lexical"] = matched_attention.pair.map(lexical_counts)
pair_attention_quality = quality_free_matched.merge(
    matched_attention, on="pair", validate="one_to_one"
)
assert (pair_attention_quality.n_scored == pair_attention_quality.n_generation_total).all()
assert (pair_attention_quality.n_attention_eligible <= pair_attention_quality.n_generation_total).all()
assert (pair_attention_quality.n_attention_lexical <= pair_attention_quality.n_attention_eligible).all()
_atomic_to_csv(pair_attention_quality,
               os.path.join(ROOT_DIR, "manifests", "pair_attention_quality_matched.csv"))
print("Saved matched NLLB spBLEU/chrF quality and attention summaries.")


# %% original code cell 22; notebook cell 30
# ============================================================
# 16. Bounded exploratory NLLB beam-3 audit
# ============================================================
beam_audit_rows, beam_audit_quality_rows = [], []
if RUN_BEAM_AUDIT:
    for p_idx, prow in pair_df.iterrows():
        pair_data = get_pair_data(prow).iloc[:BEAM_AUDIT_N].copy()
        hypotheses, generated_counts, cap_flags = [], [], []
        unknown_counts, eos_flags = [], []
        for start in range(0, BEAM_AUDIT_N, BEAM_BATCH_SIZE):
            block = pair_data.iloc[start:start + BEAM_BATCH_SIZE]
            configure_languages(prow.src_lang, prow.tgt_lang)
            tokens = tokenizer(
                block.source.astype(str).tolist(), return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LENGTH, return_attention_mask=True,
            ).to(DEVICE)
            with torch.inference_mode():
                output_ids = model.generate(
                    **tokens, forced_bos_token_id=language_id(prow.tgt_lang),
                    use_cache=True, min_length=0, max_length=BEAM_MAX_LENGTH,
                    num_beams=BEAM_NUM_BEAMS, num_return_sequences=1, do_sample=False,
                )
            sequences = output_ids.detach().cpu().tolist()
            hypotheses.extend(tokenizer.batch_decode(
                sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ))
            for sequence in sequences:
                nonpad = [int(x) for x in sequence
                          if tokenizer.pad_token_id is None
                          or int(x) != int(tokenizer.pad_token_id)]
                assert len(nonpad) >= 2
                assert nonpad[0] == int(model.config.decoder_start_token_id)
                assert nonpad[1] == language_id(prow.tgt_lang)
                count = max(0, len(nonpad) - 1)
                generated_counts.append(count)
                ended_eos = int(nonpad[-1] in _eos_ids())
                cap_flags.append(int(not ended_eos and len(nonpad) >= BEAM_MAX_LENGTH))
                unknown_counts.append(sum(
                    int(x) == int(tokenizer.unk_token_id) for x in nonpad[1:]
                ))
                eos_flags.append(ended_eos)
        assert len(hypotheses) == BEAM_AUDIT_N
        pair_rows = []
        for row, hypothesis, count, cap, unknown_count, ended_eos in zip(
            pair_data.itertuples(index=False), hypotheses, generated_counts,
            cap_flags, unknown_counts, eos_flags
        ):
            reference_subwords = reference_subword_counts[(prow.pair, int(row.run_sentence_id))]
            diag = hypothesis_diagnostics(
                hypothesis, row.target, prow.tgt_lang,
                generated_subwords=max(0, count - 1 - ended_eos),
                reference_subwords=reference_subwords, generated_steps=count,
            )
            pair_rows.append({
                "pair": prow.pair, "sentence_id": int(row.run_sentence_id),
                "corpus_sentence_id": str(row.sentence_id), "hyp_text": hypothesis,
                "reference": str(row.target), "generated_token_count": count,
                "generation_hit_max_length": cap,
                "generation_ended_with_eos": ended_eos,
                "generated_unknown_tokens": unknown_count, **diag,
            })
        beam_part = pd.DataFrame(pair_rows)
        beam_audit_rows.append(beam_part)
        beam_bleu, beam_chrf, beam_chrfpp = score_flores(
            beam_part.hyp_text.tolist(), beam_part.reference.tolist()
        )
        greedy = FREE_MATCHED[FREE_MATCHED.pair == prow.pair].sort_values(
            "sentence_id"
        ).iloc[:BEAM_AUDIT_N]
        greedy_bleu, greedy_chrf, greedy_chrfpp = score_flores(
            greedy.hyp_text.tolist(), greedy.reference.tolist()
        )
        beam_problem = (
            (beam_part.placeholder_pattern_count > 0)
            | (beam_part.max_bigram_share > 0.50)
            | (beam_part.near_generation_cap == 1)
            | (beam_part.generation_hit_max_length == 1)
            | ((beam_part.longest_character_run >= 8)
               & (beam_part.longest_character_run_ratio > 0.20))
            | (beam_part.generated_unknown_tokens > 0)
        )
        beam_audit_quality_rows.append({
            "pair": prow.pair, "relation": prow.relation, "n": BEAM_AUDIT_N,
            "num_beams": BEAM_NUM_BEAMS, "max_length": BEAM_MAX_LENGTH,
            "beam_bleu": beam_bleu, "beam_spbleu": beam_bleu,
            "beam_chrf": beam_chrf, "beam_chrfpp": beam_chrfpp,
            "greedy_bleu_same_ids": greedy_bleu,
            "greedy_spbleu_same_ids": greedy_bleu,
            "greedy_chrf_same_ids": greedy_chrf,
            "greedy_chrfpp_same_ids": greedy_chrfpp,
            "beam_minus_greedy_chrfpp": beam_chrfpp - greedy_chrfpp,
            "beam_quality_flag_rate": float(beam_problem.mean()),
            "beam_cap_hits": int(beam_part.generation_hit_max_length.sum()),
        })
        print(f"[{p_idx+1}/40] {prow.pair}: beam-3 chrF++={beam_chrfpp:.2f}, "
              f"matched greedy={greedy_chrfpp:.2f}", flush=True)
    beam_audit = pd.concat(beam_audit_rows, ignore_index=True)
    quality_beam_audit = pd.DataFrame(beam_audit_quality_rows)
    _atomic_to_csv(beam_audit, os.path.join(ROOT_DIR, "manifests", "beam_audit_hypotheses.csv"))
    _atomic_to_csv(quality_beam_audit, os.path.join(ROOT_DIR, "manifests", "quality_beam_audit.csv"))
    sync_relative_paths(["manifests/beam_audit_hypotheses.csv",
                         "manifests/quality_beam_audit.csv"])
else:
    beam_audit = pd.DataFrame()
    quality_beam_audit = pd.DataFrame()
    print("Beam audit disabled.")


# %% original code cell 23; notebook cell 31
# ============================================================
# 17. Final NLLB protocol and artifact gate
# ============================================================
assert PREFLIGHT_PASSED
assert SMOKE_TRANSLATION_PASSED and EXTRACTION_SMOKE_PASSED
assert COLLECTION_VERIFIED and TF_BATCH_EQUIVALENCE_PASSED
assert PREFIX_ASSUMPTION_PASSED and FREE_SMOKE_GENERATION_CAP_HITS == 0
validate_collected_ids(TF, range(N_SENTENCES), "TF")
validate_collected_ids(FREE, FREE_SENTENCE_IDS, "FREE")
assert len(TF) == len(pair_df) * N_SENTENCES
assert len(FREE) == len(pair_df) * FREE_N_SENTENCES
assert unresolved.empty and len(quality_free_matched) == 40

def _audit_nllb_tokens(frame, name):
    known_tags = set(pair_df.src_lang).union(pair_df.tgt_lang)
    disallowed_control_ids = (
        set(int(x) for x in tokenizer.all_special_ids)
        - {int(tokenizer.unk_token_id)}
    )
    for row in frame.itertuples(index=False):
        source_ids = json.loads(row.source_token_ids_json)
        target_ids = json.loads(row.target_token_ids_json)
        stored_source = json.loads(row.source_tokens_json)
        stored_target = json.loads(row.target_tokens_json)
        assert token_pieces(source_ids) == stored_source, f"{name}: source token mismatch"
        assert token_pieces(target_ids) == stored_target, f"{name}: target token mismatch"
        assert not known_tags.intersection(stored_source), f"{name}: source tag survived"
        assert not known_tags.intersection(stored_target), f"{name}: target tag survived"
        assert not disallowed_control_ids.intersection(map(int, source_ids)), (
            f"{name}: an NLLB source control ID survived control masking"
        )
        assert not disallowed_control_ids.intersection(map(int, target_ids)), (
            f"{name}: an NLLB target control ID survived control masking"
        )
        assert int(row.source_unknown_tokens) == sum(
            int(x) == int(tokenizer.unk_token_id) for x in source_ids
        ), f"{name}: source unknown-token count mismatch"
        assert int(row.target_unknown_tokens) == sum(
            int(x) == int(tokenizer.unk_token_id) for x in target_ids
        ), f"{name}: target unknown-token count mismatch"
    print(f"NLLB token audit passed: {name} ({len(frame)} rows)")

_audit_nllb_tokens(TF, "teacher_forced")
_audit_nllb_tokens(FREE, "free_generation")
assert (TF.source_tags_removed.astype(int) == 1).all()
assert (FREE.source_tags_removed.astype(int) == 1).all()
assert (TF.source_special_tokens_removed.astype(int) == 2).all()
assert (FREE.source_special_tokens_removed.astype(int) == 2).all()
assert (TF.target_row_alignment == "predicted_label_token").all()
assert (FREE.target_row_alignment == "predicted_generated_token").all()
assert (TF.target_special_tokens_removed.astype(int) == 2).all()
normal_free = FREE.loc[FREE.generation_hit_max_length.astype(int).eq(0)].copy()
capped_free = FREE.loc[FREE.generation_hit_max_length.astype(int).eq(1)].copy()
assert (normal_free.target_special_tokens_removed.astype(int) == 2).all()
assert (normal_free.generation_ended_with_eos.astype(int) == 1).all()
assert (normal_free.attention_inference_eligible.astype(int) == 1).all()
if len(capped_free):
    assert (capped_free.target_special_tokens_removed.astype(int) == 1).all()
    assert (capped_free.generation_ended_with_eos.astype(int) == 0).all()
    assert (capped_free.attention_inference_eligible.astype(int) == 0).all()
    assert (capped_free.attention_tensor_role == "diagnostic_capped_excluded").all()
    assert capped_free[attention_columns_to_exclude].isna().all().all()
assert len(normal_free) + len(capped_free) == len(FREE)

required_artifacts = [
    os.path.join(ROOT_DIR, "teacher_forced_all_stats.csv"),
    os.path.join(ROOT_DIR, "free_generation_all_stats.csv"),
    os.path.join(ROOT_DIR, "manifests", "collection_verification.json"),
    os.path.join(ROOT_DIR, "manifests", "free_hypothesis_diagnostics.csv"),
    os.path.join(ROOT_DIR, "manifests", "quality_free_matched.csv"),
    os.path.join(ROOT_DIR, "manifests", "sentence_quality_greedy.csv"),
    os.path.join(ROOT_DIR, "manifests", "run_manifest.json"),
    os.path.join(ROOT_DIR, "manifests", "sample_identity.csv"),
    os.path.join(ROOT_DIR, "manifests", "pair_attention_quality_matched.csv"),
    os.path.join(ROOT_DIR, "manifests", "metric_signatures.json"),
    os.path.join(ROOT_DIR, "manifests", "common_cross_model_quality.csv"),
    os.path.join(ROOT_DIR, "manifests", "unknown_token_audit.csv"),
    os.path.join(ROOT_DIR, "manifests", "unknown_token_audit_unique.csv"),
    os.path.join(ROOT_DIR, "manifests", "generation_cap_audit.csv"),
    os.path.join(ROOT_DIR, "logs", "failure_audit.csv"),
]
if RUN_BEAM_AUDIT:
    required_artifacts.extend([
        os.path.join(ROOT_DIR, "manifests", "beam_audit_hypotheses.csv"),
        os.path.join(ROOT_DIR, "manifests", "quality_beam_audit.csv"),
    ])
missing_artifacts = [path for path in required_artifacts if not os.path.isfile(path)]
assert not missing_artifacts, missing_artifacts

submission_manifest = {
    **run_manifest,
    "metric_signatures": METRIC_SIGNATURES,
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "teacher_forced_rows": len(TF), "free_generation_rows": len(FREE),
    "matched_quality_pairs": len(quality_free_matched),
    "beam_audit_enabled": RUN_BEAM_AUDIT,
    "beam_audit_pairs": len(quality_beam_audit),
    "resolved_historical_failures": int((failure_audit.status == "resolved").sum()),
    "unresolved_failures": len(unresolved),
    "checkpoint_rows_pruned": len(checkpoint_repairs),
    "free_generation_cap_hits": len(capped_free),
    "free_attention_inference_eligible_rows": len(normal_free),
    "generated_unknown_token_rows": int((FREE.target_unknown_tokens > 0).sum()),
    "beam_cap_hits": int(quality_beam_audit.beam_cap_hits.sum()) if RUN_BEAM_AUDIT else 0,
    "runtime_gates_passed": True,
}
manifest_path = os.path.join(ROOT_DIR, "manifests", "submission_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(submission_manifest, handle, indent=2, ensure_ascii=False)
sync_control_files()
print("FINAL NLLB RUN VERIFIED", {
    "root": ROOT_DIR,
    "teacher_forced_rows": len(TF),
    "free_generation_rows": len(FREE),
    "pairs": len(quality_free_matched),
    "beam_audit_pairs": len(quality_beam_audit),
    "unresolved_failures": len(unresolved),
    "sample_identity_sha256": SAMPLE_IDENTITY_SHA256,
    "submission_manifest": manifest_path,
})


# %% original code cell 24; notebook cell 32
# Optional backup: run this cell after collection and again after evaluation.
# It never deletes the local experiment. Destination is specific to ROOT_DIR.
BACKUP_TO_DRIVE = True  # Set True when ready to copy results.
if BACKUP_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    source = Path(ROOT_DIR).resolve()
    destination = Path("/content/drive/MyDrive/indic_40_backups") / source.name
    assert source.is_dir() and (source / "manifests/run_config.json").is_file()
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-r", "--info=progress2", str(source) + "/", str(destination) + "/"], check=True)
    # Full content checksum verification; slower than size-only verification.
    check = subprocess.run(["rsync", "-rcni", str(source) + "/", str(destination) + "/"],
                           check=True, capture_output=True, text=True)
    assert not check.stdout.strip(), f"Backup differs from source:\n{check.stdout}"
    print("Backup content verified:", destination)
else:
    print("Backup is optional. Set BACKUP_TO_DRIVE=True to copy the current NLLB ROOT_DIR.")

