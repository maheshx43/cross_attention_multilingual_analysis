# Original notebook code, in its original cell order.
# Requires the original Colab/IPython environment and inputs. See README.md.

# %% original code cell 0; notebook cell 2
# 0. Fresh root, dependency installation and immutable source lock
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = "/content/indic_40_balanced_holdout_750_rows250_999_locked_v4"
AUTO_DRIVE_BACKUP = True
DRIVE_SYNC_EVERY_PAIRS = 2
DRIVE_BACKUP_BASE = Path("/content/drive/MyDrive/indic_40_backups")
PILOT_ROOT_NAME = "indic_40_balanced_pilot_250_protocol_v3"
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
        subprocess.run(["rsync", "-r", "--checksum", str(DRIVE_DESTINATION) + "/",
                        str(Path(ROOT_DIR)) + "/"], check=True)
        print("Restored incremental checkpoint from", DRIVE_DESTINATION)
else:
    DRIVE_DESTINATION = None
Path(ROOT_DIR, "manifests").mkdir(parents=True, exist_ok=True)
SOURCE_LOCK_PATH = Path(ROOT_DIR, "manifests", "source_lock.json")
assert AUTO_DRIVE_BACKUP, "The confirmatory holdout requires the pilot Drive source lock."
PILOT_SOURCE_LOCK_PATH = (DRIVE_BACKUP_BASE / PILOT_ROOT_NAME / "manifests" / "source_lock.json")
PILOT_SUBMISSION_MANIFEST_PATH = (DRIVE_BACKUP_BASE / PILOT_ROOT_NAME / "manifests" / "submission_manifest.json")
PILOT_SAMPLE_IDENTITY_PATH = (DRIVE_BACKUP_BASE / PILOT_ROOT_NAME / "manifests" / "sample_identity.csv")
assert PILOT_SOURCE_LOCK_PATH.is_file(), (
    f"Pilot source lock not found: {PILOT_SOURCE_LOCK_PATH}. "
    "Do not resolve remote main for a confirmatory run."
)
assert PILOT_SUBMISSION_MANIFEST_PATH.is_file(), f"Missing pilot manifest: {PILOT_SUBMISSION_MANIFEST_PATH}"
assert PILOT_SAMPLE_IDENTITY_PATH.is_file(), f"Missing pilot identity table: {PILOT_SAMPLE_IDENTITY_PATH}"
PILOT_SOURCE_LOCK = json.loads(PILOT_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
PILOT_SUBMISSION_MANIFEST = json.loads(PILOT_SUBMISSION_MANIFEST_PATH.read_text(encoding="utf-8"))
assert PILOT_SUBMISSION_MANIFEST.get("runtime_gates_passed") is True
assert PILOT_SUBMISSION_MANIFEST.get("source_lock") == PILOT_SOURCE_LOCK, (
    "Pilot submission manifest and source_lock.json disagree."
)
required_lock_prefixes = ("git:", "hf:model:", "hf:dataset:")
missing_lock_types = [prefix for prefix in required_lock_prefixes
                      if not any(str(key).startswith(prefix) for key in PILOT_SOURCE_LOCK)]
assert not missing_lock_types, f"Pilot source lock is incomplete: {missing_lock_types}"
if SOURCE_LOCK_PATH.exists():
    existing_holdout_lock = json.loads(SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    assert existing_holdout_lock == PILOT_SOURCE_LOCK, (
        "Holdout source lock differs from the pilot. Use the new locked ROOT_DIR; "
        "never continue a holdout that resolved remote main independently."
    )
else:
    SOURCE_LOCK_PATH.write_text(
        json.dumps(PILOT_SOURCE_LOCK, indent=2), encoding="utf-8"
    )
SOURCE_LOCK = dict(PILOT_SOURCE_LOCK)
PILOT_SOURCE_LOCK_SHA256 = hashlib.sha256(
    PILOT_SOURCE_LOCK_PATH.read_bytes()
).hexdigest()
PILOT_SUBMISSION_MANIFEST_SHA256 = hashlib.sha256(
    PILOT_SUBMISSION_MANIFEST_PATH.read_bytes()
).hexdigest()
PILOT_SAMPLE_IDENTITY_SHA256 = hashlib.sha256(
    PILOT_SAMPLE_IDENTITY_PATH.read_bytes()
).hexdigest()
assert PILOT_SAMPLE_IDENTITY_SHA256 == PILOT_SUBMISSION_MANIFEST.get("sample_identity_sha256"), (
    "Pilot sample_identity.csv does not match its verified submission manifest."
)
print("Pilot source lock copied and enforced:", PILOT_SOURCE_LOCK_PATH)

def save_source_lock():
    temporary = SOURCE_LOCK_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(SOURCE_LOCK, indent=2), encoding="utf-8")
    temporary.replace(SOURCE_LOCK_PATH)

# For an independent reproduction, copy source_lock.json into the new root first.
# The first run resolves remote revisions once; subsequent runs use the saved SHAs.
def locked_repository(name, url, repo_path):
    key = "git:" + name
    if key not in SOURCE_LOCK:
        sha = subprocess.check_output(
            ["git", "ls-remote", url, "HEAD"], text=True
        ).split()[0]
        assert len(sha) == 40
        SOURCE_LOCK[key] = sha
        save_source_lock()
    sha = SOURCE_LOCK[key]
    repo = Path(repo_path)
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", url], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", sha], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", sha], check=True)
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert actual == sha, f"Unexpected revision in {repo}"
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip(), f"Tracked dependency source modified: {repo}"
    return str(repo)

INDICTRANS2_DIR = locked_repository(
    "IndicTrans2", "https://github.com/AI4Bharat/IndicTrans2.git",
    Path(ROOT_DIR, "dependencies", "IndicTrans2"))
INDICNLP_LIBRARY_DIR = locked_repository(
    "indic_nlp_library", "https://github.com/anoopkunchukuttan/indic_nlp_library.git",
    Path(INDICTRANS2_DIR, "indic_nlp_library"))
INDICNLP_RESOURCES_DIR = locked_repository(
    "indic_nlp_resources", "https://github.com/anoopkunchukuttan/indic_nlp_resources.git",
    Path(INDICTRANS2_DIR, "indic_nlp_resources"))
os.environ["INDIC_RESOURCES_PATH"] = INDICNLP_RESOURCES_DIR

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
    "--upgrade-strategy", "only-if-needed",
    "transformers==4.51.3", "tokenizers==0.21.1", "huggingface_hub==0.30.2",
    "datasets==4.0.0", "sacrebleu==2.3.1", "sentencepiece", "sacremoses",
    INDICNLP_LIBRARY_DIR, "indictranstoolkit==1.1.1"], check=True)
print("Dependencies installed; git revisions locked in", SOURCE_LOCK_PATH)


# %% original code cell 1; notebook cell 3
# ============================================================
# 1. Imports, deterministic controls, and run configuration
# ============================================================
import gc
import platform
import re
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import unicodedata

import numpy as np
import pandas as pd
import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

SEED = 42
PIPELINE_SCHEMA_VERSION = "3.0.0"
PIPELINE_SOURCE_SHA256 = "d207f8d0a0f0292a32b365deac6b33857f4d4661006afd7baaaa3338dac5f17a"

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
MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"
MODEL_REVISION = "main"
DATASET_NAME = "openlanguagedata/flores_plus"
DATASET_SPLIT = "devtest"
DATASET_REVISION = "main"

CORPUS_ROW_START = 250
CORPUS_ROW_STOP_EXCLUSIVE = 1000
N_SENTENCES = CORPUS_ROW_STOP_EXCLUSIVE - CORPUS_ROW_START
FREE_N_SENTENCES = N_SENTENCES
FREE_SENTENCE_IDS = tuple(range(FREE_N_SENTENCES))
SELECTION_ROLE = "confirmatory_holdout"
MAX_LENGTH = 256                 # encoder and teacher-forced label gate
FREE_MAX_LENGTH = 256            # total decoder length, matching model card/config
assert 0 < FREE_N_SENTENCES <= N_SENTENCES

DATA_MODE = "flores_pivot"       # holdout protocol is frozen to FLORES+ pivot mode
assert DATA_MODE == "flores_pivot", "Confirmatory holdout selection is defined only for FLORES+."
DIRECT_IDS_ARE_SHARED = False
DIRECT_PARALLEL_MANIFEST = None
RUN_MODE = "both"                # both | teacher_forced_only | free_generation_only
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

# Non-primary exploratory audit; intentionally three beams, not the model-card five.
RUN_BEAM_AUDIT = True
BEAM_NUM_BEAMS = 3
BEAM_MAX_LENGTH = 256
BEAM_AUDIT_N = min(50, N_SENTENCES)
BEAM_BATCH_SIZE = 8

DISK_SAFETY_FACTOR = 1.25
DISK_FIXED_OVERHEAD_GB = 10
MIN_FREE_VRAM_GB = 8
STRICT_PILOT_ENVIRONMENT_MATCH = True

# Confirmatory execution uses the same recorded software and GPU class as the pilot.
PILOT_PACKAGES = PILOT_SUBMISSION_MANIFEST.get("packages", {})
ENVIRONMENT_PACKAGES = [
    "transformers", "tokenizers", "datasets", "huggingface-hub", "torch",
    "sacrebleu", "numpy", "pandas", "pyarrow", "sentencepiece", "sacremoses",
    "indic-nlp-library", "indictranstoolkit",
]
CURRENT_PACKAGES = {}
for package_name in ENVIRONMENT_PACKAGES:
    try:
        CURRENT_PACKAGES[package_name] = version(package_name)
    except PackageNotFoundError:
        CURRENT_PACKAGES[package_name] = None
environment_differences = {
    name: {"pilot": PILOT_PACKAGES.get(name), "holdout": CURRENT_PACKAGES.get(name)}
    for name in ENVIRONMENT_PACKAGES
    if PILOT_PACKAGES.get(name) != CURRENT_PACKAGES.get(name)
}
pilot_python = PILOT_SUBMISSION_MANIFEST.get("python")
pilot_cuda = PILOT_SUBMISSION_MANIFEST.get("torch_cuda")
pilot_gpu = PILOT_SUBMISSION_MANIFEST.get("gpu")
current_gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else None
if STRICT_PILOT_ENVIRONMENT_MATCH:
    assert not environment_differences, f"Pilot/holdout package mismatch: {environment_differences}"
    assert platform.python_version() == pilot_python, (platform.python_version(), pilot_python)
    assert torch.version.cuda == pilot_cuda, (torch.version.cuda, pilot_cuda)
    assert current_gpu == pilot_gpu, (current_gpu, pilot_gpu)
print("Pilot environment match verified:", {
    "python": pilot_python, "torch_cuda": pilot_cuda, "gpu": pilot_gpu,
    "package_differences": environment_differences,
})

for sub in ("teacher_forced", "free_generation", "logs", "manifests"):
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
    "pilot_root_name": PILOT_ROOT_NAME,
    "pilot_source_lock_sha256": PILOT_SOURCE_LOCK_SHA256,
    "pilot_submission_manifest_sha256": PILOT_SUBMISSION_MANIFEST_SHA256,
    "strict_pilot_environment_match": STRICT_PILOT_ENVIRONMENT_MATCH,
    "corpus_row_start": CORPUS_ROW_START,
    "corpus_row_stop_exclusive": CORPUS_ROW_STOP_EXCLUSIVE,
    "data_mode": DATA_MODE,
    "direct_parallel_manifest": DIRECT_PARALLEL_MANIFEST,
    "direct_ids_are_shared": DIRECT_IDS_ARE_SHARED,
    "n_sentences": N_SENTENCES,
    "free_n_sentences": FREE_N_SENTENCES,
    "free_sentence_ids": list(FREE_SENTENCE_IDS),
    "max_length": MAX_LENGTH,
    "free_max_length": FREE_MAX_LENGTH,
    "batch_sizes": {"tf": BATCH_SIZE_TF, "free": BATCH_SIZE_FREE},
    "save_full_tensors_free": SAVE_FULL_TENSORS_FREE,
    "greedy_num_beams": GREEDY_NUM_BEAMS,
    "greedy_min_length": 0,
    "greedy_num_return_sequences": 1,
    "official_generation_deviations": [
        "primary attention generation is greedy (num_beams=1, not model-card num_beams=5)",
        "exploratory beam audit uses num_beams=3, not model-card num_beams=5",
    ],
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
}
config_path = os.path.join(ROOT_DIR, "manifests", "run_config.json")
if os.path.isfile(config_path):
    with open(config_path, encoding="utf-8") as handle:
        previous_config = json.load(handle)
    assert previous_config == CORE_CONFIG, (
        "ROOT_DIR contains an incompatible protocol/configuration. Use the matching "
        "checkpoint or a new ROOT_DIR; never mix schemas."
    )
else:
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(CORE_CONFIG, handle, indent=2, ensure_ascii=False)

print("Device:", DEVICE, "| dtype:", DTYPE)
print("Root:", ROOT_DIR)
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
RESOLVED_DATASET_COMMIT = (locked_hf_revision(DATASET_NAME, DATASET_REVISION, "dataset")
                           if DATA_MODE == "flores_pivot" else None)


# %% original code cell 3; notebook cell 5
from IndicTransToolkit.processor import IndicProcessor

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, token=HF_TOKEN, revision=RESOLVED_MODEL_COMMIT
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE,
    attn_implementation="eager", token=HF_TOKEN, revision=RESOLVED_MODEL_COMMIT,
).to(DEVICE).eval()

def new_inference_processor():
    """Batch-local processor; its FIFO entity maps must be consumed immediately."""
    return IndicProcessor(inference=True)

def new_training_processor():
    """No entity-map queue; used only for teacher-forced target preprocessing."""
    return IndicProcessor(inference=False)

assert hasattr(tokenizer, "src_decoder") and hasattr(tokenizer, "tgt_decoder"), (
    "Pinned IndicTrans tokenizer must expose distinct source/target vocabularies."
)
print("model_max_length:", tokenizer.model_max_length)
print("Tokenizer padding sides:", tokenizer.padding_side,
      getattr(tokenizer, "_target_tokenizer", None))
print("Model loaded on", DEVICE, "dtype", next(model.parameters()).dtype)
assert getattr(model.config, "_commit_hash", None) == RESOLVED_MODEL_COMMIT
print("Resolved model commit:", RESOLVED_MODEL_COMMIT)


# %% original code cell 4; notebook cell 6
# ============================================================
# 3. Explicit-vocabulary tokenizer probes
# ============================================================
_PROBE_TEXT_RAW = "यह एक परीक्षण वाक्य है ।"
_PROBE_SRC, _PROBE_TGT = "hin_Deva", "kan_Knda"

def src_pieces(ids):
    ids = [int(i) for i in ids]
    missing = [i for i in ids if i not in tokenizer.src_decoder]
    assert not missing, f"IDs absent from source vocabulary: {missing[:5]}"
    return [tokenizer.src_decoder[i] for i in ids]

def tgt_pieces(ids):
    ids = [int(i) for i in ids]
    missing = [i for i in ids if i not in tokenizer.tgt_decoder]
    assert not missing, f"IDs absent from target vocabulary: {missing[:5]}"
    return [tokenizer.tgt_decoder[i] for i in ids]

def tokenizer_state_probe(tag=""):
    proc = new_inference_processor()
    processed = proc.preprocess_batch([_PROBE_TEXT_RAW], src_lang=_PROBE_SRC, tgt_lang=_PROBE_TGT)
    encoded = tokenizer(processed, add_special_tokens=True, return_attention_mask=True)
    ids, mask = encoded["input_ids"][0], encoded["attention_mask"][0]
    active = [i for i, m in enumerate(mask) if m]
    pieces = src_pieces([ids[i] for i in active])
    assert pieces[:2] == [_PROBE_SRC, _PROBE_TGT]
    print(f"[probe @ {tag}] active-prefix={pieces[:2]} ids={tuple(ids[i] for i in active[:8])}")
    return tuple(ids[i] for i in active)

_probe_baseline = tokenizer_state_probe("baseline")


# %% original code cell 5; notebook cell 7
# ============================================================
# 3b. Processor isolation and official inference smoke test
# ============================================================
PROBE_TGT_TEXT = "ಇದು ಒಂದು ಪರೀಕ್ಷಾ ವಾಕ್ಯ."

# Training-style target preprocessing must not create placeholder FIFO state.
tgt_processor = new_training_processor()
tgt_processed = tgt_processor.preprocess_batch(
    [PROBE_TGT_TEXT], src_lang=_PROBE_TGT, tgt_lang=_PROBE_SRC, is_target=True
)
tgt_encoded = tokenizer(text_target=tgt_processed, add_special_tokens=True)
tgt_ids_probe = tgt_encoded["input_ids"][0]
assert len(tgt_ids_probe) >= 3
print("Target pieces from target vocabulary:", tgt_pieces(tgt_ids_probe)[:10])

# A fresh inference processor owns exactly this source batch and immediately
# consumes exactly its own placeholder maps during postprocessing.
smoke_ip = new_inference_processor()
processed = smoke_ip.preprocess_batch([_PROBE_TEXT_RAW], src_lang=_PROBE_SRC, tgt_lang=_PROBE_TGT)
inputs = tokenizer(processed, truncation=True, padding="longest",
                   return_tensors="pt", return_attention_mask=True).to(DEVICE)
seed_everything(SEED)
with torch.inference_mode():
    generated = model.generate(
        **inputs, use_cache=True, min_length=0,
        max_length=FREE_MAX_LENGTH, num_beams=1,
        num_return_sequences=1, do_sample=False,
    )
decoded = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=True)
translations = smoke_ip.postprocess_batch(decoded, lang=_PROBE_TGT)
assert translations[0].strip()
assert tokenizer_state_probe("after target and generation") == _probe_baseline
SMOKE_TRANSLATION_PASSED = True
print("Smoke translation:", translations[0])
print("Processor isolation and explicit-vocabulary checks passed.")


# %% original code cell 6; notebook cell 8
# 4. Load the selected data mode
ds = None
DATASET_FINGERPRINT = None
if DATA_MODE == "flores_pivot":
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=RESOLVED_DATASET_COMMIT)
    DATASET_FINGERPRINT = getattr(ds, "_fingerprint", None)
    print(ds)
    print("Dataset fingerprint:", DATASET_FINGERPRINT)
else:
    print("Direct-parallel mode: references come exclusively from the supplied manifest.")


# %% original code cell 7; notebook cell 9
# 4b. Mode-specific data tables
TARGET_LANGS = {
    "hin_Deva", "ben_Beng", "mar_Deva", "guj_Gujr", "pan_Guru", "ory_Orya",
    "asm_Beng", "tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym",
}
ID_COL, LANG_COL, SCRIPT_COL, TEXT_COL = "id", "iso_639_3", "iso_15924", "text"
if DATA_MODE == "flores_pivot":
    df_long = ds.to_pandas()
    df_long["lang_code"] = df_long[LANG_COL].astype(str) + "_" + df_long[SCRIPT_COL].astype(str)
    df_indic = df_long[df_long["lang_code"].isin(TARGET_LANGS)].copy()
    dupes = df_indic[df_indic.duplicated(subset=[ID_COL, "lang_code"], keep=False)]
    assert len(dupes) == 0, "Duplicate FLORES id/language rows found."
    df_wide = df_indic.pivot(index=ID_COL, columns="lang_code", values=TEXT_COL).reset_index()
    missing = sorted(TARGET_LANGS - set(df_wide.columns))
    assert not missing, f"Missing required languages: {missing}"
    assert CORPUS_ROW_START >= 0
    assert CORPUS_ROW_STOP_EXCLUSIVE <= len(df_wide), (
        f"Holdout stop {CORPUS_ROW_STOP_EXCLUSIVE} exceeds {len(df_wide)} available FLORES rows."
    )
    assert CORPUS_ROW_STOP_EXCLUSIVE - CORPUS_ROW_START == N_SENTENCES
    df_wide_all = df_wide.sort_values(ID_COL).reset_index(drop=True)
    df_wide = df_wide_all.iloc[CORPUS_ROW_START:CORPUS_ROW_STOP_EXCLUSIVE].reset_index(drop=True)
    assert len(df_wide) == N_SENTENCES
    print("FLORES confirmatory holdout table:", df_wide.shape,
          "sorted row positions:", (CORPUS_ROW_START, CORPUS_ROW_STOP_EXCLUSIVE), flush=True)


assert DATA_MODE in {"flores_pivot", "direct_parallel"}
if DATA_MODE == "direct_parallel":
    assert DIRECT_PARALLEL_MANIFEST, "Set DIRECT_PARALLEL_MANIFEST for direct_parallel mode."
    if str(DIRECT_PARALLEL_MANIFEST).lower().endswith(".parquet"):
        direct_parallel = pd.read_parquet(DIRECT_PARALLEL_MANIFEST)
    else:
        direct_parallel = pd.read_csv(DIRECT_PARALLEL_MANIFEST)
    required = {"pair", "sentence_id", "source", "target"}
    assert required.issubset(direct_parallel.columns), (
        f"Direct manifest must have {required}; got {list(direct_parallel.columns)}"
    )
    assert not direct_parallel.duplicated(["pair", "sentence_id"]).any(), (
        "Direct manifest has duplicate pair/sentence_id rows."
    )
    print("DIRECT PARALLEL mode enabled:", direct_parallel.shape, flush=True)
else:
    direct_parallel = None
    print("FLORES-PIVOT mode enabled: use only for shared-meaning/descriptive claims.", flush=True)


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
    if DATA_MODE == "direct_parallel":
        frame = direct_parallel.loc[direct_parallel.pair == pair,
                                    ["sentence_id", "source", "target"]].copy()
        frame = frame.sort_values("sentence_id").head(N_SENTENCES).reset_index(drop=True)
        assert len(frame) == N_SENTENCES, f"{pair}: direct manifest has {len(frame)}, need {N_SENTENCES}."
    else:
        frame = pd.DataFrame({
            "sentence_id": df_wide[ID_COL].iloc[:N_SENTENCES].to_numpy(),
            "source": df_wide[prow.src_lang].iloc[:N_SENTENCES].to_numpy(),
            "target": df_wide[prow.tgt_lang].iloc[:N_SENTENCES].to_numpy(),
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
    if DATA_MODE == "flores_pivot" or DIRECT_IDS_ARE_SHARED:
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

# Prove that this run is a genuine sentence-level holdout from the frozen pilot.
pilot_identity = pd.read_csv(
    PILOT_SAMPLE_IDENTITY_PATH, dtype={"corpus_sentence_id": str}
)
required_identity_columns = {
    "pair", "sentence_id", "corpus_sentence_id", "source_sha256",
    "target_sha256", "in_free_subset",
}
assert required_identity_columns.issubset(pilot_identity.columns)
assert not pilot_identity.duplicated(["pair", "sentence_id"]).any()
assert set(pilot_identity.pair) == set(pair_df.pair)
pilot_selected = pilot_identity[pilot_identity.in_free_subset.astype(int) == 1].copy()
holdout_selected = sample_identity[sample_identity.in_free_subset.astype(int) == 1].copy()
pilot_ids = set(pilot_selected.corpus_sentence_id.astype(str))
holdout_ids = set(holdout_selected.corpus_sentence_id.astype(str))
assert len(pilot_selected) == len(pair_df) * 250
assert len(holdout_selected) == len(pair_df) * N_SENTENCES
assert len(pilot_ids) == 250, f"Pilot does not contain 250 shared meanings: {len(pilot_ids)}"
assert len(holdout_ids) == 750, f"Holdout does not contain 750 shared meanings: {len(holdout_ids)}"
overlap_ids = sorted(pilot_ids.intersection(holdout_ids))
assert not overlap_ids, f"Pilot/holdout corpus overlap: {overlap_ids[:10]}"
HOLDOUT_SELECTION_AUDIT = {
    "selection_role": SELECTION_ROLE,
    "pilot_root_name": PILOT_ROOT_NAME,
    "pilot_sample_identity_sha256": PILOT_SAMPLE_IDENTITY_SHA256,
    "holdout_sample_identity_sha256": SAMPLE_IDENTITY_SHA256,
    "pilot_unique_meanings": len(pilot_ids),
    "holdout_unique_meanings": len(holdout_ids),
    "corpus_id_overlap_count": len(overlap_ids),
    "corpus_row_start": CORPUS_ROW_START,
    "corpus_row_stop_exclusive": CORPUS_ROW_STOP_EXCLUSIVE,
    "direction_count": len(pair_df),
}
holdout_audit_path = os.path.join(
    ROOT_DIR, "manifests", "holdout_selection_audit.json"
)
with open(holdout_audit_path, "w", encoding="utf-8") as handle:
    json.dump(HOLDOUT_SELECTION_AUDIT, handle, indent=2)
print("Pilot/holdout zero-overlap audit passed:", HOLDOUT_SELECTION_AUDIT)

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
# 6. Disk, VRAM, and no-truncation preflight gates
# ============================================================
disk = shutil.disk_usage("/content")
free_disk_gb = disk.free / 1024**3
print(f"Free Colab disk: {free_disk_gb:.1f} GB")
if DEVICE == "cuda":
    free_vram, total_vram = torch.cuda.mem_get_info()
    free_vram_gb, total_vram_gb = free_vram / 1024**3, total_vram / 1024**3
    print(f"GPU VRAM: {free_vram_gb:.1f}/{total_vram_gb:.1f} GB free")
    assert free_vram_gb >= MIN_FREE_VRAM_GB, (
        f"Need at least {MIN_FREE_VRAM_GB} GB free VRAM; found {free_vram_gb:.1f} GB."
    )
else:
    print("CPU runtime detected; the full extraction run will be impractically slow.")

def encoded_lengths(texts, src_lang, tgt_lang, target=False):
    if target:
        proc = new_training_processor()
        processed = proc.preprocess_batch(
            texts, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
        )
        encoded = tokenizer(text_target=processed, add_special_tokens=True, truncation=False)
    else:
        proc = new_inference_processor()
        processed = proc.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        encoded = tokenizer(processed, add_special_tokens=True, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]

decoder_module = getattr(getattr(model, "model", None), "decoder", None)
ATTENTION_LAYERS = int(getattr(model.config, "decoder_layers", 0) or len(decoder_module.layers))
ATTENTION_HEADS = int(getattr(model.config, "decoder_attention_heads", 0)
                      or decoder_module.layers[0].encoder_attn.num_heads)
assert ATTENTION_LAYERS > 0 and ATTENTION_HEADS > 0
FLOAT16_BYTES = np.dtype(np.float16).itemsize
NPY_HEADER_ALLOWANCE = 256
tf_tensor_bytes_upper = 0
free_tensor_bytes_reference = 0
free_tensor_bytes_cap_upper = 0
free_id_set = set(FREE_SENTENCE_IDS)

truncation_violations = []
max_source_tokens = 0
max_target_tokens = 0
for _, prow in pair_df.iterrows():
    pair_data = get_pair_data(prow)
    for start in range(0, len(pair_data), 128):
        block = pair_data.iloc[start:start + 128]
        source_lengths = encoded_lengths(
            block.source.tolist(), prow.src_lang, prow.tgt_lang, target=False
        )
        target_lengths = encoded_lengths(
            block.target.tolist(), prow.src_lang, prow.tgt_lang, target=True
        )
        max_source_tokens = max(max_source_tokens, max(source_lengths))
        max_target_tokens = max(max_target_tokens, max(target_lengths))
        for offset, (source_length, target_length) in enumerate(zip(source_lengths, target_lengths)):
            # Saved tensors exclude control/special tokens, so full token lengths
            # provide a conservative upper estimate.
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
                    ATTENTION_LAYERS * ATTENTION_HEADS * source_length * (FREE_MAX_LENGTH - 1)
                    * FLOAT16_BYTES + NPY_HEADER_ALLOWANCE
                )
        for offset, length in enumerate(source_lengths):
            if length > MAX_LENGTH:
                truncation_violations.append((prow.pair, start + offset, "source", length))
        for offset, length in enumerate(target_lengths):
            if length > MAX_LENGTH:
                truncation_violations.append((prow.pair, start + offset, "target", length))

print("Maximum untruncated tokens:", {"source": max_source_tokens, "target": max_target_tokens})
tensor_reference_gib = (tf_tensor_bytes_upper + free_tensor_bytes_reference) / 1024**3
tensor_cap_upper_gib = (tf_tensor_bytes_upper + free_tensor_bytes_cap_upper) / 1024**3
required_disk_gib = tensor_cap_upper_gib * DISK_SAFETY_FACTOR + DISK_FIXED_OVERHEAD_GB
print("Tensor storage estimate (GiB):", {
    "reference_length_estimate": round(tensor_reference_gib, 2),
    "generation_cap_upper": round(tensor_cap_upper_gib, 2),
    "required_with_margin": round(required_disk_gib, 2),
})
assert free_disk_gb >= required_disk_gib, (
    f"Frozen workload needs about {required_disk_gib:.1f} GiB free using the "
    f"generation-cap upper bound and safety margin; found {free_disk_gb:.1f} GiB."
)

assert not truncation_violations, (
    f"Found {len(truncation_violations)} inputs longer than MAX_LENGTH={MAX_LENGTH}; "
    f"first examples: {truncation_violations[:10]}"
)
PREFLIGHT_PASSED = True
print("PRE-FLIGHT PASSED: disk, VRAM, and input/target truncation gates.")

PREFLIGHT_REPORT = {
    "free_disk_gib": free_disk_gb,
    "free_vram_gib": free_vram_gb if DEVICE == "cuda" else None,
    "max_source_tokens": max_source_tokens, "max_target_tokens": max_target_tokens,
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


# %% original code cell 10; notebook cell 12
# ============================================================
# 6. Isolated preprocessing for one teacher-forced observation
# ============================================================
def prepare_pair(src_lang, tgt_lang, src_text, tgt_text):
    # Inference preprocessing makes the encoder input identical in semantics to
    # free generation. The instance is discarded, so its queued entity map can
    # never contaminate another batch.
    src_ip = new_inference_processor()
    src_proc = src_ip.preprocess_batch([src_text], src_lang=src_lang, tgt_lang=tgt_lang)
    src_tok = tokenizer(src_proc, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH,
                        return_attention_mask=True)

    # Labels use training/teacher-forcing preprocessing: no entity placeholders,
    # no queue, and target-side tokenization.
    tgt_ip = new_training_processor()
    tgt_proc = tgt_ip.preprocess_batch(
        [tgt_text], src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
    )
    tgt_tok = tokenizer(text_target=tgt_proc, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH,
                        add_special_tokens=True, return_attention_mask=True)
    labels = tgt_tok["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    tgt_tok["labels"] = labels
    return src_proc[0], tgt_proc[0], src_tok, tgt_tok


# %% original code cell 11; notebook cell 13
# ============================================================
# 7. Real-sentence source/target vocabulary sanity check
# ============================================================
demo = pair_df.iloc[0]
demo_data = get_pair_data(demo).iloc[0]
src_proc0, tgt_proc0, src_tok0, tgt_tok0 = prepare_pair(
    demo.src_lang, demo.tgt_lang, demo_data.source, demo_data.target
)
src_ids0 = src_tok0["input_ids"][0].tolist()
tgt_ids0 = tgt_tok0["input_ids"][0].tolist()
src_active0 = [i for i, m in enumerate(src_tok0["attention_mask"][0].tolist()) if m]
tgt_active0 = [i for i, m in enumerate(tgt_tok0["attention_mask"][0].tolist()) if m]
src_active_pieces0 = src_pieces([src_ids0[i] for i in src_active0])
tgt_active_pieces0 = tgt_pieces([tgt_ids0[i] for i in tgt_active0])
assert src_active_pieces0[:2] == [demo.src_lang, demo.tgt_lang]
assert _probe_baseline == tokenizer_state_probe("after prepare_pair")
print("Source active prefix:", src_active_pieces0[:2])
print("Target-vocabulary pieces:", tgt_active_pieces0[:12])
print("Tokenizer/processor sanity check passed.")


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
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
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
# 11b. Unified extraction with correct padding, vocabularies and processor state
# ============================================================
def _special_ids():
    return set(int(x) for x in tokenizer.all_special_ids)

def _is_punctuation_piece(piece):
    text = str(piece).replace("▁", "").replace("Ġ", "").strip()
    if not text:
        return True
    return all(unicodedata.category(ch)[0] in {"P", "S", "Z", "C"} for ch in text)

def _active_positions(input_ids, attention_mask):
    assert len(input_ids) == len(attention_mask)
    return [i for i, m in enumerate(attention_mask) if int(m) == 1]

def _kept_source_positions(input_ids, attention_mask, src_lang, tgt_lang):
    """Remove the two active prefix tags after accounting for left padding."""
    active = _active_positions(input_ids, attention_mask)
    assert len(active) >= 3, f"Unexpectedly short active source sequence: {active}"
    active_pieces = src_pieces([input_ids[i] for i in active])
    assert active_pieces[:2] == [src_lang, tgt_lang], (
        f"Wrong active source prefix: expected {[src_lang, tgt_lang]}, "
        f"observed {active_pieces[:2]}"
    )
    special = _special_ids()
    kept = [i for i in active[2:] if int(input_ids[i]) not in special]
    return kept, active[:2]

def _kept_target_positions(input_ids, attention_mask=None):
    active = (range(len(input_ids)) if attention_mask is None
              else _active_positions(input_ids, attention_mask))
    special = _special_ids()
    return [i for i in active if int(input_ids[i]) not in special]

def _prepare_teacher_forced_batch(src_lang, tgt_lang, src_texts, tgt_texts):
    # Batch-local inference processor: encoder preprocessing matches generation,
    # while its unconsumed placeholder queue dies with this local instance.
    src_ip = new_inference_processor()
    src_proc = src_ip.preprocess_batch(src_texts, src_lang=src_lang, tgt_lang=tgt_lang)
    src_tok = tokenizer(src_proc, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH,
                        return_attention_mask=True)

    # Target labels follow training semantics and never enter the inference FIFO.
    tgt_ip = new_training_processor()
    tgt_proc = tgt_ip.preprocess_batch(
        tgt_texts, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
    )
    tgt_tok = tokenizer(text_target=tgt_proc, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LENGTH,
                        add_special_tokens=True, return_attention_mask=True)
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

def _summaries_with_token_controls(full, source_ids, target_ids):
    source_tokens = src_pieces(source_ids)
    target_tokens = tgt_pieces(target_ids)
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
        "source_tags_removed": 2,
        "src_retained_tokens": len(source_tokens),
        "tgt_retained_tokens": len(target_tokens),
        "src_lexical_tokens": len(src_lex),
        "tgt_lexical_tokens": len(tgt_lex),
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
    with torch.inference_mode():
        out = model(
            input_ids=src_tok["input_ids"].to(DEVICE),
            attention_mask=src_tok["attention_mask"].to(DEVICE),
            labels=labels.to(DEVICE), output_attentions=True, return_dict=True,
        )
    stacked = torch.stack([x.float().cpu() for x in out.cross_attentions], dim=0).numpy()
    del out
    results = []
    for b in range(len(src_texts)):
        source_ids = src_tok["input_ids"][b].tolist()
        source_mask = src_tok["attention_mask"][b].tolist()
        target_ids = tgt_tok["input_ids"][b].tolist()
        target_mask = tgt_tok["attention_mask"][b].tolist()
        src_keep, tag_positions = _kept_source_positions(
            source_ids, source_mask, src_lang, tgt_lang
        )
        tgt_keep = _kept_target_positions(target_ids, target_mask)
        if not src_keep or not tgt_keep:
            results.append(None)
            continue
        full = stacked[:, b][:, :, tgt_keep, :][:, :, :, src_keep]
        kept_src_ids = [source_ids[i] for i in src_keep]
        kept_tgt_ids = [target_ids[i] for i in tgt_keep]
        stat = _summaries_with_token_controls(full, kept_src_ids, kept_tgt_ids)
        stat["source_tag_positions_json"] = json.dumps(tag_positions)
        results.append((full, stat))
    return results

def _eos_ids():
    values = [getattr(tokenizer, "eos_token_id", None),
              getattr(model.generation_config, "eos_token_id", None)]
    result = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(x) for x in value)
        else:
            result.add(int(value))
    return result

def extract_attention_free_batch(src_lang, tgt_lang, src_texts,
                                 max_length=FREE_MAX_LENGTH):
    # This processor is paired one-to-one with this batch's postprocess call.
    generation_ip = new_inference_processor()
    src_proc = generation_ip.preprocess_batch(src_texts, src_lang=src_lang, tgt_lang=tgt_lang)
    src_tok = tokenizer(src_proc, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_attention_mask=True)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=src_tok["input_ids"].to(DEVICE),
            attention_mask=src_tok["attention_mask"].to(DEVICE),
            num_beams=GREEDY_NUM_BEAMS, num_return_sequences=1,
            min_length=0, do_sample=DO_SAMPLE, use_cache=True,
            max_length=max_length, output_attentions=True,
            return_dict_in_generate=True,
        )
    if not getattr(generated, "cross_attentions", None):
        raise RuntimeError("Generation returned no cross-attention tensors.")
    n_steps = len(generated.cross_attentions)
    n_layers = len(generated.cross_attentions[0])
    layers = [
        torch.cat([generated.cross_attentions[s][layer].float().cpu()
                   for s in range(n_steps)], dim=2)
        for layer in range(n_layers)
    ]
    stacked = torch.stack(layers, dim=0).numpy()
    eos_ids = _eos_ids()
    decoded_raw, bookkeeping = [], []

    for b in range(len(src_texts)):
        sequence = generated.sequences[b].detach().cpu().tolist()
        prefix_len = len(sequence) - n_steps
        assert prefix_len >= 1, (len(sequence), n_steps)
        generated_ids = [int(x) for x in sequence[prefix_len:]]
        active_steps = n_steps
        for j, token_id in enumerate(generated_ids):
            if token_id in eos_ids:
                active_steps = j + 1
                break
        active_ids = generated_ids[:active_steps]
        decoded_raw.append(tokenizer.decode(active_ids, skip_special_tokens=True))
        bookkeeping.append((generated_ids, active_ids, active_steps))

    hypotheses = generation_ip.postprocess_batch(decoded_raw, lang=tgt_lang)
    assert len(hypotheses) == len(src_texts)
    results = []
    for b, hypothesis in enumerate(hypotheses):
        generated_ids, active_ids, active_steps = bookkeeping[b]
        source_ids = src_tok["input_ids"][b].tolist()
        source_mask = src_tok["attention_mask"][b].tolist()
        src_keep, tag_positions = _kept_source_positions(
            source_ids, source_mask, src_lang, tgt_lang
        )
        tgt_keep = _kept_target_positions(active_ids)
        if not src_keep or not tgt_keep:
            results.append(None)
            continue
        full = stacked[:, b][:, :, tgt_keep, :][:, :, :, src_keep]
        kept_src_ids = [source_ids[i] for i in src_keep]
        kept_tgt_ids = [active_ids[i] for i in tgt_keep]
        stat = _summaries_with_token_controls(full, kept_src_ids, kept_tgt_ids)
        ended_with_eos = int(bool(active_ids) and active_ids[-1] in eos_ids)
        reached_length_boundary = int((1 + active_steps) >= max_length)
        stat.update({
            "source_tag_positions_json": json.dumps(tag_positions),
            "generated_steps": int(active_steps),
            "generation_ended_with_eos": ended_with_eos,
            # Conservatively flag EOS in the final allowed slot too: it may have
            # been forced by the length processor rather than naturally emitted.
            "generation_hit_max_length": reached_length_boundary,
        })
        results.append((full, stat, hypothesis))
    del generated
    return results

def probe_source_prefix(prow):
    source = get_pair_data(prow).iloc[0].source
    proc = new_inference_processor()
    processed = proc.preprocess_batch([source], src_lang=prow.src_lang, tgt_lang=prow.tgt_lang)[0]
    parts = processed.split(maxsplit=2)
    assert len(parts) == 3 and parts[:2] == [prow.src_lang, prow.tgt_lang]
    encoded = tokenizer(processed, add_special_tokens=True, return_attention_mask=True)
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    keep, tag_positions = _kept_source_positions(ids, mask, prow.src_lang, prow.tgt_lang)
    tag_tokens = src_pieces([ids[i] for i in tag_positions])
    assert tag_tokens == [prow.src_lang, prow.tgt_lang]
    assert not set(tag_tokens).intersection(src_pieces([ids[i] for i in keep]))
    return {"pair": prow.pair, "prefix_text": parts[:2],
            "prefix_positions": tag_positions, "prefix_tokens": tag_tokens}

PREFIX_PROBE_RECORDS = [probe_source_prefix(prow) for _, prow in pair_df.iterrows()]
PREFIX_ASSUMPTION_PASSED = len(PREFIX_PROBE_RECORDS) == len(pair_df)
print("Verified active (padding-aware) two-token prefix for all 40 directions.")

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
_free_smoke = extract_attention_free_batch(_smoke.src_lang, _smoke.tgt_lang, [_srcs[0]])[0]
assert _free_smoke is not None and _free_smoke[0].ndim == 4
assert np.isfinite(_free_smoke[0]).all() and _free_smoke[2].strip()
assert _free_smoke[1]["generation_hit_max_length"] == 0
assert _free_smoke[1]["source_tags_removed"] == 2
FREE_SMOKE_GENERATION_CAP_HITS = int(_free_smoke[1]["generation_hit_max_length"])
EXTRACTION_SMOKE_PASSED = True
TF_BATCH_EQUIVALENCE_PASSED = True
print("Production smoke passed: padding-aware tags, target vocab, TF batch equivalence, greedy FIFO.")

# The 25-row smoke sample would otherwise miss previously observed pathological
# IDs. Probe them without adding them to the fixed 25-row scientific sample.
EDGE_CASE_SMOKE_REPORT = []
if N_SENTENCES == 25 and DATA_MODE == "flores_pivot":
    edge_cases = [
        ("hin_Deva__ory_Orya", 82),
        ("asm_Beng__kan_Knda", 174),
        ("ben_Beng__tam_Taml", 243),
    ]
    for pair_name, positional_id in edge_cases:
        prow = pair_df.loc[pair_df.pair == pair_name].iloc[0]
        source_text = df_wide_all.iloc[positional_id][prow.src_lang]
        seed_everything(SEED + int(prow.name))
        edge = extract_attention_free_batch(prow.src_lang, prow.tgt_lang, [source_text])[0]
        assert edge is not None
        edge_tensor, edge_stats, edge_hypothesis = edge
        words = edge_hypothesis.split()
        unique_ratio = len(set(words)) / len(words) if words else 0.0
        record = {
            "pair": pair_name, "positional_id": positional_id,
            "generated_steps": int(edge_stats["generated_steps"]),
            "hit_max_length": int(edge_stats["generation_hit_max_length"]),
            "word_count": len(words), "unique_word_ratio": unique_ratio,
            "hypothesis_preview": edge_hypothesis[:240],
        }
        EDGE_CASE_SMOKE_REPORT.append(record)
        print("EDGE CASE:", record)
        del edge_tensor, edge
    assert not any(row["hit_max_length"] for row in EDGE_CASE_SMOKE_REPORT), (
        "A known edge-case probe reached total max_length=256. Do not start the "
        "larger run until its decoding policy is reviewed."
    )

timing_n = 8
timing_pairs = np.linspace(0, len(pair_df) - 1, timing_n, dtype=int)
timing_sentences = np.linspace(0, N_SENTENCES - 1, timing_n, dtype=int)
timing_start = time.perf_counter()
with tempfile.TemporaryDirectory() as timing_dir:
    for j, (pair_index, sentence_id) in enumerate(zip(timing_pairs, timing_sentences)):
        prow = pair_df.iloc[int(pair_index)]
        text = get_pair_data(prow).iloc[int(sentence_id)].source
        seed_everything(SEED + int(pair_index))
        result = extract_attention_free_batch(prow.src_lang, prow.tgt_lang, [text])[0]
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
del _tf_smoke, _free_smoke, _batch_src, _batch_tgt
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %% original code cell 16; notebook cell 22
# Reproducibility manifest written before collection and checked on resume
package_names = ["transformers", "tokenizers", "datasets", "huggingface-hub", "torch",
                 "sacrebleu", "numpy", "pandas", "pyarrow", "sentencepiece", "sacremoses",
                 "indic-nlp-library", "indictranstoolkit"]
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
    "packages": package_versions, "python": platform.python_version(),
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if DEVICE == "cuda" else None,
    "dtype": str(DTYPE), "pair_list": pair_df.to_dict("records"),
    "preflight": PREFLIGHT_REPORT,
    "validation_observations": {
        "fp16_fp32_mean_max_diff": FP16_FP32_MEAN_MAX_DIFF,
        "fp16_fp32_overall_max_diff": FP16_FP32_OVERALL_MAX_DIFF,
        "repeat_run_max_diff": REPEAT_RUN_MAX_DIFF,
        "bit_identical": BIT_IDENTICAL,
        "two_prefix_tokens_verified": PREFIX_ASSUMPTION_PASSED,
        "tf_batch_equivalence_passed": TF_BATCH_EQUIVALENCE_PASSED,
        "free_smoke_generation_cap_hits": FREE_SMOKE_GENERATION_CAP_HITS,
        "edge_case_smoke_report": EDGE_CASE_SMOKE_REPORT,
    },
    "free_timing_probe": FREE_TIMING_REPORT,
    "processor_protocol": "batch-local inference processor; inference=False teacher-forced targets",
    "token_vocabulary_protocol": "src_decoder for encoder IDs; tgt_decoder for label/generated IDs",
    "metric_protocol": "IndicTrans2 preprocessing; SacreBLEU tokenize=none; chrF order=6 beta=2; chrF++ word_order=2",
    "free_subset_policy": "fixed positional IDs; no substitution; strict unresolved-failure gate",
    "shared_content_across_pairs": DATA_MODE == "flores_pivot" or DIRECT_IDS_ARE_SHARED,
    "runtime_gates_passed": False,
}
run_manifest_path = os.path.join(ROOT_DIR, "manifests", "run_manifest.json")
if os.path.isfile(run_manifest_path):
    with open(run_manifest_path) as handle:
        previous = json.load(handle)
    for key in ("config", "source_lock", "sample_identity_sha256", "pipeline_schema_version", "pipeline_source_sha256", "packages", "python", "torch_cuda", "dtype"):
        assert previous[key] == run_manifest[key], f"Resume provenance changed: {key}"
    # Preserve original provenance; hardware changes are explicit in session records.
else:
    with open(run_manifest_path, "w") as handle:
        json.dump(run_manifest, handle, indent=2)
sessions = os.path.join(ROOT_DIR, "manifests", "runtime_sessions.jsonl")
with open(sessions, "a") as handle:
    handle.write(json.dumps(run_manifest) + "\n")
print("Pre-run manifest and session provenance saved.")


# %% original code cell 17; notebook cell 23
# ============================================================
# 12. Main collection loop with tensor-aware resume reconciliation
# ============================================================
tf_dir = os.path.join(ROOT_DIR, "teacher_forced")
free_dir = os.path.join(ROOT_DIR, "free_generation")
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
    clean = data.drop_duplicates("sentence_id", keep="last").sort_values("sentence_id")
    _atomic_to_csv(clean, path)

def _tensor_is_valid(path):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) <= 128:
            return False
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return array.ndim == 4 and all(int(x) > 0 for x in array.shape)
    except Exception:
        return False

REPAIR_COLUMNS = ["utc", "pair", "mode", "sentence_id", "reason", "checkpoint"]
if os.path.isfile(repair_path) and os.path.getsize(repair_path):
    checkpoint_repairs = pd.read_csv(repair_path).to_dict("records")
else:
    checkpoint_repairs = []

def _record_repairs(rows):
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        checkpoint_repairs.append({"utc": now, **row})
    _atomic_to_csv(pd.DataFrame(checkpoint_repairs, columns=REPAIR_COLUMNS), repair_path)

COMMON_REQUIRED = {
    "pair", "sentence_id", "corpus_sentence_id", "pipeline_schema_version",
    "source_tags_removed", "source_token_ids_json", "target_token_ids_json",
    "source_tokens_json", "target_tokens_json", "diagonal_excess_10pct",
    "expected_normalized_displacement",
}

def _load_checkpoint(path, tensor_dir, pair_name, mode, expected_ids, require_tensor):
    if not (RESUME and os.path.isfile(path)):
        return pd.DataFrame()
    try:
        data = pd.read_csv(path, dtype={"corpus_sentence_id": str})
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()
    repairs = []
    if not COMMON_REQUIRED.issubset(data.columns):
        for sid in data.get("sentence_id", pd.Series(dtype=int)).tolist():
            repairs.append({"pair": pair_name, "mode": mode, "sentence_id": sid,
                            "reason": "incompatible_or_legacy_schema", "checkpoint": path})
        _record_repairs(repairs)
        _atomic_to_csv(pd.DataFrame(), path)
        return pd.DataFrame()
    if mode == "free_generation" and not {
        "hyp_text", "generated_steps", "generation_hit_max_length"
    }.issubset(data.columns):
        for sid in data.sentence_id.tolist():
            repairs.append({"pair": pair_name, "mode": mode, "sentence_id": sid,
                            "reason": "missing_free_generation_columns", "checkpoint": path})
        _record_repairs(repairs)
        _atomic_to_csv(pd.DataFrame(), path)
        return pd.DataFrame()

    data["sentence_id"] = pd.to_numeric(data["sentence_id"], errors="coerce")
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
            elif str(row["pipeline_schema_version"]) != PIPELINE_SCHEMA_VERSION:
                reason = "schema_version_mismatch"
            elif int(row["source_tags_removed"]) != 2:
                reason = "source_tag_audit_failed"
            elif require_tensor and not _tensor_is_valid(os.path.join(tensor_dir, f"{sid}.npy")):
                reason = "missing_or_invalid_tensor"
        if reason is None:
            valid_rows.append(index)
            seen.add(sid)
        else:
            repairs.append({"pair": pair_name, "mode": mode, "sentence_id": sid,
                            "reason": reason, "checkpoint": path})
    clean = data.loc[sorted(valid_rows)].copy()
    if not clean.empty:
        clean["sentence_id"] = clean["sentence_id"].astype(int)
        clean = clean.sort_values("sentence_id")
    if repairs:
        _record_repairs(repairs)
        _atomic_to_csv(clean, path)
        print(f"Resume repaired {len(repairs)} invalid row(s): {pair_name}/{mode}", flush=True)
    return clean

def _count(data):
    return int(data.sentence_id.nunique()) if "sentence_id" in data.columns else 0

FAILURE_COLUMNS = ["pair", "sentence_id", "mode", "error"]
if RESUME and os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
    try:
        failures = pd.read_csv(log_path).to_dict("records")
    except pd.errors.EmptyDataError:
        failures = []
else:
    failures = []

def _write_failures():
    frame = pd.DataFrame(failures, columns=FAILURE_COLUMNS).drop_duplicates()
    _atomic_to_csv(frame, log_path)
    return frame

def sync_relative_paths(relative_paths):
    if not AUTO_DRIVE_BACKUP:
        return
    assert DRIVE_DESTINATION is not None
    assert Path("/content/drive/MyDrive").is_dir(), "Google Drive is not mounted"
    for relative_path in sorted(set(relative_paths)):
        source_path = Path(ROOT_DIR, relative_path)
        if not source_path.exists():
            continue
        destination = Path(DRIVE_DESTINATION, relative_path)
        if source_path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            command = ["rsync", "-r", "--checksum", "--partial",
                       str(source_path) + "/", str(destination) + "/"]
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = ["rsync", "--checksum", "--partial", str(source_path), str(destination)]
        subprocess.run(command, check=True, capture_output=True, text=True)

def sync_control_files():
    relative_paths = []
    for folder in ("manifests", "logs"):
        directory = Path(ROOT_DIR, folder)
        if directory.is_dir():
            relative_paths.extend(str(item.relative_to(ROOT_DIR)) for item in directory.iterdir()
                                  if item.is_file())
    sync_relative_paths(relative_paths)

def sync_pair_checkpoints(pair_names):
    paths = []
    for pair_name in pair_names:
        paths.extend([f"teacher_forced/{pair_name}", f"teacher_forced/{pair_name}_stats.csv",
                      f"free_generation/{pair_name}", f"free_generation/{pair_name}_stats.csv"])
    sync_relative_paths(paths)
    sync_control_files()
    print(f"Drive checkpoint synced for {len(pair_names)} pair(s): {pair_names}", flush=True)

pending_drive_pairs, all_tf, all_free = [], [], []
for p_idx, prow in pair_df.iterrows():
    seed_everything(SEED + p_idx)
    pair_name, pair_data = prow.pair, get_pair_data(prow)
    tf_csv = os.path.join(tf_dir, f"{pair_name}_stats.csv")
    free_csv = os.path.join(free_dir, f"{pair_name}_stats.csv")
    tf_tensor_dir = os.path.join(tf_dir, pair_name)
    free_tensor_dir = os.path.join(free_dir, pair_name)
    os.makedirs(tf_tensor_dir, exist_ok=True)
    os.makedirs(free_tensor_dir, exist_ok=True)
    tf_pair = _load_checkpoint(tf_csv, tf_tensor_dir, pair_name, "teacher_forced",
                               range(N_SENTENCES), True)
    free_pair = _load_checkpoint(free_csv, free_tensor_dir, pair_name, "free_generation",
                                 FREE_SENTENCE_IDS, SAVE_FULL_TENSORS_FREE)
    tf_done = set(tf_pair.get("sentence_id", pd.Series(dtype=int)).astype(int))
    free_done = set(free_pair.get("sentence_id", pd.Series(dtype=int)).astype(int))
    print(f"\n[{p_idx+1}/{len(pair_df)}] {pair_name}: TF={len(tf_done)}/{N_SENTENCES}, "
          f"FREE={len(free_done)}/{N_FREE}", flush=True)

    if RUN_TEACHER_FORCED:
        missing = [sid for sid in range(N_SENTENCES) if sid not in tf_done]
        for batch_idx, start in enumerate(range(0, len(missing), BATCH_SIZE_TF), 1):
            sids = missing[start:start + BATCH_SIZE_TF]
            srcs = pair_data.iloc[sids].source.tolist()
            tgts = pair_data.iloc[sids].target.tolist()
            show = batch_idx == 1 or batch_idx % PRINT_EVERY_TF_BATCHES == 0 or start + len(sids) == len(missing)
            if show:
                print(f"[{p_idx+1}/40] {pair_name} TF {sids[0]}..{sids[-1]} START", flush=True)
            try:
                results = extract_attention_teacher_forced_batch(
                    prow.src_lang, prow.tgt_lang, srcs, tgts
                )
            except Exception as exc:
                print(f"TF batch failed: {exc}; retrying individual fixed IDs", flush=True)
                results = []
                for sid, src, tgt in zip(sids, srcs, tgts):
                    try:
                        results.append(extract_attention_teacher_forced_batch(
                            prow.src_lang, prow.tgt_lang, [src], [tgt]
                        )[0])
                    except Exception as item_exc:
                        failures.append({"pair": pair_name, "sentence_id": sid,
                                         "mode": "teacher_forced", "error": str(item_exc)})
                        results.append(None)
            for sid, result in zip(sids, results):
                if result is None:
                    failures.append({"pair": pair_name, "sentence_id": sid,
                                     "mode": "teacher_forced_empty", "error": "No retained tokens"})
                    continue
                full, stat = result
                assert np.isfinite(full).all()
                _atomic_save_npy(full.astype(np.float16), os.path.join(tf_tensor_dir, f"{sid}.npy"))
                stat.update({"pair": pair_name, "relation": prow.relation,
                             "family_relation": prow.family_relation, "sentence_id": sid,
                             "corpus_sentence_id": str(pair_data.iloc[sid].sentence_id)})
                tf_pair = pd.concat([tf_pair, pd.DataFrame([stat])], ignore_index=True)
            _write_checkpoint(tf_pair, tf_csv)
            _write_failures()
            if show:
                print(f"[{p_idx+1}/40] {pair_name} TF {_count(tf_pair)}/{N_SENTENCES} DONE", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if RUN_FREE_GENERATION:
        seed_everything(SEED + p_idx)
        missing = [sid for sid in FREE_SENTENCE_IDS if sid not in free_done]
        for batch_idx, start in enumerate(range(0, len(missing), BATCH_SIZE_FREE), 1):
            sids = missing[start:start + BATCH_SIZE_FREE]
            srcs = pair_data.iloc[sids].source.tolist()
            try:
                results = extract_attention_free_batch(prow.src_lang, prow.tgt_lang, srcs)
            except Exception as exc:
                for sid in sids:
                    failures.append({"pair": pair_name, "sentence_id": sid,
                                     "mode": "free_generation", "error": str(exc)})
                results = [None] * len(sids)
            for sid, src, result in zip(sids, srcs, results):
                if result is None:
                    try:
                        result = extract_attention_free_batch(
                            prow.src_lang, prow.tgt_lang, [src]
                        )[0]
                    except Exception as exc:
                        failures.append({"pair": pair_name, "sentence_id": sid,
                                         "mode": "free_generation", "error": str(exc)})
                if result is None:
                    failures.append({"pair": pair_name, "sentence_id": sid,
                                     "mode": "free_generation_empty", "error": "No retained tokens"})
                    continue
                full, stat, hypothesis = result
                assert np.isfinite(full).all()
                if int(stat["generation_hit_max_length"]):
                    failures.append({"pair": pair_name, "sentence_id": sid,
                                     "mode": "free_generation_cap",
                                     "error": f"Reached total FREE_MAX_LENGTH={FREE_MAX_LENGTH}"})
                    _write_failures()
                    raise RuntimeError(
                        f"{pair_name} sentence {sid} reached total FREE_MAX_LENGTH={FREE_MAX_LENGTH}; "
                        "do not retain a truncated attention observation."
                    )
                if SAVE_FULL_TENSORS_FREE:
                    _atomic_save_npy(full.astype(np.float16),
                                     os.path.join(free_tensor_dir, f"{sid}.npy"))
                stat.update({"pair": pair_name, "relation": prow.relation,
                             "family_relation": prow.family_relation, "sentence_id": sid,
                             "corpus_sentence_id": str(pair_data.iloc[sid].sentence_id),
                             "hyp_text": hypothesis})
                free_pair = pd.concat([free_pair, pd.DataFrame([stat])], ignore_index=True)
            _write_checkpoint(free_pair, free_csv)
            _write_failures()
            if batch_idx == 1 or batch_idx % PRINT_EVERY_FREE_ITEMS == 0 or start + len(sids) == len(missing):
                print(f"[{p_idx+1}/40] {pair_name} FREE: {_count(free_pair)}/{N_FREE}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Reconcile once more before this pair is declared complete or copied to Drive.
    tf_pair = _load_checkpoint(tf_csv, tf_tensor_dir, pair_name, "teacher_forced",
                               range(N_SENTENCES), True)
    free_pair = _load_checkpoint(free_csv, free_tensor_dir, pair_name, "free_generation",
                                 FREE_SENTENCE_IDS, SAVE_FULL_TENSORS_FREE)
    all_tf.append(tf_pair)
    all_free.append(free_pair)
    _write_failures()
    pending_drive_pairs.append(pair_name)
    if AUTO_DRIVE_BACKUP and (len(pending_drive_pairs) >= DRIVE_SYNC_EVERY_PAIRS
                              or p_idx == len(pair_df) - 1):
        sync_pair_checkpoints(pending_drive_pairs)
        pending_drive_pairs.clear()

TF = pd.concat(all_tf, ignore_index=True)
FREE = pd.concat(all_free, ignore_index=True)
tf_all_path = os.path.join(ROOT_DIR, "teacher_forced_all_stats.csv")
free_all_path = os.path.join(ROOT_DIR, "free_generation_all_stats.csv")
_atomic_to_csv(TF, tf_all_path)
_atomic_to_csv(FREE, free_all_path)
failure_df = _write_failures()

def reconcile_failures(failure_frame, tf_frame, free_frame):
    done_tf = set(zip(tf_frame.get("pair", []), tf_frame.get("sentence_id", [])))
    done_free = set(zip(free_frame.get("pair", []), free_frame.get("sentence_id", [])))
    audit = failure_frame.copy()
    states = []
    for row in audit.itertuples(index=False):
        completed = done_tf if str(row.mode).startswith("teacher_forced") else done_free
        states.append("resolved" if (row.pair, int(row.sentence_id)) in completed else "unresolved")
    audit["status"] = states
    return audit

failure_audit = reconcile_failures(failure_df, TF, FREE)
failure_audit_path = os.path.join(ROOT_DIR, "logs", "failure_audit.csv")
_atomic_to_csv(failure_audit, failure_audit_path)
unresolved = failure_audit[failure_audit["status"] == "unresolved"]
assert unresolved.empty, f"Unresolved fixed IDs; resume: {unresolved.head(10).to_dict('records')}"

validate_collected_ids(TF, range(N_SENTENCES), "TF")
validate_collected_ids(FREE, FREE_SENTENCE_IDS, "FREE")
assert len(TF) == len(pair_df) * N_SENTENCES
assert len(FREE) == len(pair_df) * N_FREE
assert (TF["pipeline_schema_version"] == PIPELINE_SCHEMA_VERSION).all()
assert (FREE["pipeline_schema_version"] == PIPELINE_SCHEMA_VERSION).all()
assert (TF["source_tags_removed"].astype(int) == 2).all()
assert (FREE["source_tags_removed"].astype(int) == 2).all()
assert int(FREE["generation_hit_max_length"].sum()) == 0

tf_counts = TF.groupby("pair").sentence_id.nunique().reindex(pair_df.pair, fill_value=0)
free_counts = FREE.groupby("pair").sentence_id.nunique().reindex(pair_df.pair, fill_value=0)
collection_verification = {
    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
    "teacher_forced_rows": len(TF), "free_generation_rows": len(FREE),
    "pairs": len(pair_df), "teacher_forced_per_pair_min": int(tf_counts.min()),
    "teacher_forced_per_pair_max": int(tf_counts.max()),
    "free_generation_per_pair_min": int(free_counts.min()),
    "free_generation_per_pair_max": int(free_counts.max()),
    "checkpoint_rows_pruned": len(checkpoint_repairs),
    "logged_failures": len(failure_df),
    "free_generation_length_boundary_hits": int(FREE["generation_hit_max_length"].sum()),
}
with open(os.path.join(ROOT_DIR, "manifests", "collection_verification.json"), "w", encoding="utf-8") as handle:
    json.dump(collection_verification, handle, indent=2)
COLLECTION_VERIFIED = True
print("COLLECTION VERIFIED:", collection_verification, flush=True)
sync_relative_paths(["teacher_forced_all_stats.csv", "free_generation_all_stats.csv",
                     "manifests/collection_verification.json", "logs/failure_audit.csv"])
sync_control_files()


# %% original code cell 18; notebook cell 25
import sys

# ============================================================
# 13. Official IndicTrans2 metric helper and DATA_MODE-safe references
# ============================================================
PREPROCESS_SCRIPT = os.path.join(INDICTRANS2_DIR, "scripts", "preprocess_translate.py")
assert os.path.isfile(PREPROCESS_SCRIPT), f"Missing: {PREPROCESS_SCRIPT}"

def official_indictrans2_preprocess(texts, lang_code):
    texts = ["" if x is None else str(x) for x in texts]
    assert all("\n" not in x and "\r" not in x for x in texts), (
        "Metric inputs must be one physical line each; refusing to mutate them."
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.txt")
        output_path = os.path.join(tmpdir, "output.tok")
        with open(input_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(texts) + "\n")
        result = subprocess.run(
            [sys.executable, PREPROCESS_SCRIPT, input_path, output_path, lang_code, "false", "false"],
            cwd=INDICTRANS2_DIR, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"IndicTrans2 metric preprocessing failed for {lang_code}:\n{result.stderr}"
            )
        with open(output_path, encoding="utf-8") as handle:
            processed = [line.rstrip("\n") for line in handle]
    assert len(processed) == len(texts), f"Line-count change for {lang_code}"
    return processed

def score_official(hypotheses, references, target_lang):
    assert len(hypotheses) == len(references) > 0
    hyp_tok = official_indictrans2_preprocess(hypotheses, target_lang)
    ref_tok = official_indictrans2_preprocess(references, target_lang)
    bleu = sacrebleu.corpus_bleu(hyp_tok, [ref_tok], tokenize="none").score
    chrf = sacrebleu.corpus_chrf(hyp_tok, [ref_tok], word_order=0).score
    chrfpp = sacrebleu.corpus_chrf(hyp_tok, [ref_tok], word_order=2).score
    return bleu, chrf, chrfpp

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
    REFERENCE_TABLE,
    on=["pair", "sentence_id"],
    how="left",
    validate="one_to_one",
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
    return float(counts.iloc[0] / len(grams))

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
        "generated_subword_count": int(generated_subwords),
        "reference_subword_count": int(reference_subwords),
        "subword_length_ratio": (generated_subwords / reference_subwords
                                 if reference_subwords else np.nan),
        "generated_steps": int(generated_steps),
        "near_generation_cap": int((1 + generated_steps) >= 0.90 * FREE_MAX_LENGTH),
    }

# Reference subword lengths use the same inference=False target preprocessing as TF labels.
reference_subword_counts = {}
special_ids = _special_ids()
for prow in pair_df.itertuples(index=False):
    group = REFERENCE_TABLE[REFERENCE_TABLE.pair == prow.pair].sort_values("sentence_id")
    proc = new_training_processor()
    processed = proc.preprocess_batch(group.reference.astype(str).tolist(),
                                      src_lang=prow.tgt_lang, tgt_lang=prow.src_lang,
                                      is_target=True)
    encoded = tokenizer(text_target=processed, add_special_tokens=True, truncation=False)
    for sid, ids in zip(group.sentence_id.astype(int), encoded["input_ids"]):
        reference_subword_counts[(prow.pair, sid)] = sum(int(x) not in special_ids for x in ids)

diagnostic_rows = []
for row in FREE_MATCHED.itertuples(index=False):
    diag = hypothesis_diagnostics(
        row.hyp_text, row.reference, row.tgt_lang,
        generated_subwords=int(row.tgt_retained_tokens),
        reference_subwords=reference_subword_counts[(row.pair, int(row.sentence_id))],
        generated_steps=int(row.generated_steps),
    )
    diag.update({"pair": row.pair, "sentence_id": int(row.sentence_id),
                 "lexical_view_unavailable": int(row.analysis_view != "lexical_no_punctuation")})
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
    "max_trigram_share", "near_generation_cap", "lexical_view_unavailable",
    "quality_flag",
]
display(free_quality_diag.loc[free_quality_diag.quality_flag,
       [c for c in diagnostic_columns if c in free_quality_diag.columns]]
       .sort_values(["pair", "sentence_id"]))


# %% original code cell 21; notebook cell 28
# ============================================================
# 15. Matched greedy corpus metrics and sentence chrF++
# ============================================================
sentence_quality_parts, matched_rows = [], []
for p_idx, prow in pair_df.iterrows():
    group = FREE_MATCHED[FREE_MATCHED.pair == prow.pair].sort_values("sentence_id").copy()
    assert len(group) == FREE_N_SENTENCES
    hyp_tok = official_indictrans2_preprocess(group.hyp_text.tolist(), prow.tgt_lang)
    ref_tok = official_indictrans2_preprocess(group.reference.tolist(), prow.tgt_lang)
    group["sentence_chrfpp"] = [
        sacrebleu.sentence_chrf(h, [r], word_order=2).score
        for h, r in zip(hyp_tok, ref_tok)
    ]
    sentence_quality_parts.append(group[["pair", "sentence_id", "corpus_sentence_id",
                                         "sentence_chrfpp"]])
    bleu = sacrebleu.corpus_bleu(hyp_tok, [ref_tok], tokenize="none").score
    chrf = sacrebleu.corpus_chrf(hyp_tok, [ref_tok], word_order=0).score
    chrfpp = sacrebleu.corpus_chrf(hyp_tok, [ref_tok], word_order=2).score
    matched_rows.append({
        "pair": prow.pair, "relation": prow.relation,
        "family_relation": prow.family_relation, "src_lang": prow.src_lang,
        "tgt_lang": prow.tgt_lang, "bleu_greedy": bleu, "chrf_greedy": chrf,
        "chrfpp_greedy": chrfpp,
        "mean_sentence_chrfpp": float(group.sentence_chrfpp.mean()),
        "median_sentence_chrfpp": float(group.sentence_chrfpp.median()),
        "n_scored": len(group),
    })
    print(f"[{p_idx+1}/40] {prow.pair}: BLEU={bleu:.2f} chrF={chrf:.2f} chrF++={chrfpp:.2f}")

sentence_quality = pd.concat(sentence_quality_parts, ignore_index=True)
_atomic_to_csv(sentence_quality, os.path.join(ROOT_DIR, "manifests", "sentence_quality_greedy.csv"))
quality_free_matched = pd.DataFrame(matched_rows)
assert len(quality_free_matched) == 40 and quality_free_matched.pair.is_unique
assert set(quality_free_matched.pair) == set(pair_df.pair)
assert (quality_free_matched.n_scored == FREE_N_SENTENCES).all()
matched_path = os.path.join(ROOT_DIR, "manifests", "quality_free_matched.csv")
_atomic_to_csv(quality_free_matched, matched_path)

attention_columns = [
    "diagonal_mass_5pct", "diagonal_excess_5pct",
    "diagonal_mass_10pct", "diagonal_excess_10pct",
    "diagonal_mass_20pct", "diagonal_excess_20pct",
    "expected_normalized_displacement", "normalized_entropy",
    "mean_row_max", "row_max_excess", "src_len", "tgt_len",
]
validate_collected_ids(FREE_MATCHED, FREE_SENTENCE_IDS, "FREE_MATCHED")
matched_attention = FREE_MATCHED.groupby("pair")[attention_columns].mean().add_prefix(
    "free_mean_"
).reset_index()
total_counts = FREE_MATCHED.groupby("pair").size()
lexical_counts = FREE_MATCHED.groupby("pair").diagonal_excess_10pct.count()
matched_attention["n_attention_total"] = matched_attention.pair.map(total_counts)
matched_attention["n_attention_lexical"] = matched_attention.pair.map(lexical_counts)
pair_attention_quality = quality_free_matched.merge(matched_attention, on="pair",
                                                     validate="one_to_one")
assert (pair_attention_quality.n_scored == pair_attention_quality.n_attention_total).all()
assert (pair_attention_quality.n_attention_lexical <= pair_attention_quality.n_attention_total).all()
_atomic_to_csv(pair_attention_quality,
               os.path.join(ROOT_DIR, "manifests", "pair_attention_quality_matched.csv"))
print("Saved matched greedy corpus metrics, sentence chrF++, and pair attention summaries.")


# %% original code cell 22; notebook cell 30
# ============================================================
# 16. Bounded exploratory three-beam audit (not a final-gate requirement)
# ============================================================
beam_audit_rows, beam_audit_quality_rows = [], []
if RUN_BEAM_AUDIT:
    for p_idx, prow in pair_df.iterrows():
        pair_data = get_pair_data(prow).iloc[:BEAM_AUDIT_N].copy()
        hypotheses, generated_counts, cap_flags = [], [], []
        for start in range(0, BEAM_AUDIT_N, BEAM_BATCH_SIZE):
            block = pair_data.iloc[start:start + BEAM_BATCH_SIZE]
            audit_ip = new_inference_processor()
            processed = audit_ip.preprocess_batch(
                block.source.astype(str).tolist(), src_lang=prow.src_lang, tgt_lang=prow.tgt_lang
            )
            tokens = tokenizer(processed, return_tensors="pt", padding=True,
                               truncation=True, max_length=MAX_LENGTH,
                               return_attention_mask=True).to(DEVICE)
            with torch.inference_mode():
                output_ids = model.generate(
                    **tokens, use_cache=True, min_length=0, max_length=BEAM_MAX_LENGTH,
                    num_beams=BEAM_NUM_BEAMS, num_return_sequences=1, do_sample=False,
                )
            sequences = output_ids.detach().cpu().tolist()
            decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
            hypotheses.extend(audit_ip.postprocess_batch(decoded, lang=prow.tgt_lang))
            for sequence in sequences:
                nonpad = [int(x) for x in sequence
                          if tokenizer.pad_token_id is None or int(x) != int(tokenizer.pad_token_id)]
                # Encoder-decoder sequence includes the decoder-start token.
                count = max(0, len(nonpad) - 1)
                generated_counts.append(count)
                cap_flags.append(int(len(nonpad) >= BEAM_MAX_LENGTH))
        assert len(hypotheses) == BEAM_AUDIT_N
        pair_rows = []
        for i, (row, hypothesis, count, cap) in enumerate(zip(
            pair_data.itertuples(index=False), hypotheses, generated_counts, cap_flags
        )):
            reference_subwords = reference_subword_counts[(prow.pair, int(row.run_sentence_id))]
            # Text-level diagnostic subword count is approximate here; the beam audit
            # is not used for primary attention/quality inference.
            diag = hypothesis_diagnostics(hypothesis, row.target, prow.tgt_lang,
                                          generated_subwords=count,
                                          reference_subwords=reference_subwords,
                                          generated_steps=count)
            pair_rows.append({
                "pair": prow.pair, "sentence_id": int(row.run_sentence_id),
                "corpus_sentence_id": str(row.sentence_id), "hyp_text": hypothesis,
                "reference": str(row.target), "generated_token_count": count,
                "generation_hit_max_length": cap, **diag,
            })
        beam_part = pd.DataFrame(pair_rows)
        beam_audit_rows.append(beam_part)
        beam_bleu, beam_chrf, beam_chrfpp = score_official(
            beam_part.hyp_text.tolist(), beam_part.reference.tolist(), prow.tgt_lang
        )
        greedy = FREE_MATCHED[FREE_MATCHED.pair == prow.pair].sort_values("sentence_id").iloc[:BEAM_AUDIT_N]
        greedy_bleu, greedy_chrf, greedy_chrfpp = score_official(
            greedy.hyp_text.tolist(), greedy.reference.tolist(), prow.tgt_lang
        )
        beam_problem = (
            (beam_part.placeholder_pattern_count > 0)
            | (beam_part.max_bigram_share > 0.50)
            | (beam_part.near_generation_cap == 1)
        )
        beam_audit_quality_rows.append({
            "pair": prow.pair, "relation": prow.relation, "n": BEAM_AUDIT_N,
            "num_beams": BEAM_NUM_BEAMS, "max_length": BEAM_MAX_LENGTH,
            "beam_bleu": beam_bleu, "beam_chrf": beam_chrf, "beam_chrfpp": beam_chrfpp,
            "greedy_bleu_same_ids": greedy_bleu, "greedy_chrf_same_ids": greedy_chrf,
            "greedy_chrfpp_same_ids": greedy_chrfpp,
            "beam_minus_greedy_chrfpp": beam_chrfpp - greedy_chrfpp,
            "beam_quality_flag_rate": float(beam_problem.mean()),
            "beam_cap_hits": int(beam_part.generation_hit_max_length.sum()),
        })
        print(f"[{p_idx+1}/40] {prow.pair} audit: beam chrF++={beam_chrfpp:.2f}, "
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
    print("Beam audit disabled. This is expected for the primary matched-greedy protocol.")


# %% original code cell 23; notebook cell 31
# ============================================================
# 17. Final protocol and artifact gate
# ============================================================
assert PREFLIGHT_PASSED
assert SMOKE_TRANSLATION_PASSED and EXTRACTION_SMOKE_PASSED
assert COLLECTION_VERIFIED and TF_BATCH_EQUIVALENCE_PASSED
assert PREFIX_ASSUMPTION_PASSED and FREE_SMOKE_GENERATION_CAP_HITS == 0
validate_collected_ids(TF, range(N_SENTENCES), "TF")
validate_collected_ids(FREE, FREE_SENTENCE_IDS, "FREE")
assert len(TF) == len(pair_df) * N_SENTENCES
assert len(FREE) == len(pair_df) * FREE_N_SENTENCES
assert unresolved.empty
assert len(quality_free_matched) == 40

def _audit_explicit_vocabularies(frame, name):
    known_tags = set(pair_df.src_lang).union(pair_df.tgt_lang)
    for row in frame.itertuples(index=False):
        source_ids = json.loads(row.source_token_ids_json)
        target_ids = json.loads(row.target_token_ids_json)
        stored_source = json.loads(row.source_tokens_json)
        stored_target = json.loads(row.target_tokens_json)
        assert src_pieces(source_ids) == stored_source, f"{name}: source vocabulary mismatch"
        assert tgt_pieces(target_ids) == stored_target, f"{name}: target vocabulary mismatch"
        assert not known_tags.intersection(stored_source), f"{name}: source tag survived masking"
    print(f"Explicit source/target vocabulary audit passed: {name} ({len(frame)} rows)")

_audit_explicit_vocabularies(TF, "teacher_forced")
_audit_explicit_vocabularies(FREE, "free_generation")
assert not any(column.startswith("no_final_source") for column in TF.columns)
assert not any(column.startswith("no_final_source") for column in FREE.columns)
assert (TF.pipeline_schema_version == PIPELINE_SCHEMA_VERSION).all()
assert (FREE.pipeline_schema_version == PIPELINE_SCHEMA_VERSION).all()

required_artifacts = [
    os.path.join(ROOT_DIR, "teacher_forced_all_stats.csv"),
    os.path.join(ROOT_DIR, "free_generation_all_stats.csv"),
    os.path.join(ROOT_DIR, "manifests", "collection_verification.json"),
    os.path.join(ROOT_DIR, "manifests", "free_hypothesis_diagnostics.csv"),
    os.path.join(ROOT_DIR, "manifests", "quality_free_matched.csv"),
    os.path.join(ROOT_DIR, "manifests", "sentence_quality_greedy.csv"),
    os.path.join(ROOT_DIR, "manifests", "run_manifest.json"),
    os.path.join(ROOT_DIR, "manifests", "sample_identity.csv"),
    os.path.join(ROOT_DIR, "manifests", "holdout_selection_audit.json"),
    os.path.join(ROOT_DIR, "manifests", "pair_attention_quality_matched.csv"),
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
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "teacher_forced_rows": len(TF), "free_generation_rows": len(FREE),
    "matched_quality_pairs": len(quality_free_matched),
    "beam_audit_enabled": RUN_BEAM_AUDIT,
    "beam_audit_pairs": len(quality_beam_audit),
    "resolved_historical_failures": int((failure_audit.status == "resolved").sum()),
    "unresolved_failures": len(unresolved),
    "checkpoint_rows_pruned": len(checkpoint_repairs),
    "runtime_gates_passed": True,
}
manifest_path = os.path.join(ROOT_DIR, "manifests", "submission_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(submission_manifest, handle, indent=2, ensure_ascii=False)
sync_control_files()
print("FINAL RUN VERIFIED")
print(json.dumps(submission_manifest, indent=2, ensure_ascii=False))


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
    print("Backup is optional. Set BACKUP_TO_DRIVE=True to copy the current ROOT_DIR.")

