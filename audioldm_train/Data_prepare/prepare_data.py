# Databricks notebook source
"""
Data preparation script for AudioLDM training/finetuning.

This script is designed to run on Databricks. It:
1. Reads metadata from a CSV file (first column = file ID, fourth column = caption).
2. Reads corresponding .flac audio files from a mounted volume directory.
3. Converts each audio to mono WAV at 48 kHz.
4. Saves the processed .wav files to a mounted volume folder.
5. Generates a single aggregated JSON file containing all processed entries,
   following the structure expected by downstream code.

Usage (Databricks notebook or job):
    Adjust the configuration variables below, then run the script.
"""
# COMMAND ----------
import csv
import json
import os
import io
import logging
import shutil
import tempfile

import soundfile as sf
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration – adjust these to match your environment
# ---------------------------------------------------------------------------
# change the base directory to the current working directory of the SCRIPT TO ENSURE THE CSV FILES ARE FOUND
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Path to the input CSV file (on DBFS or mounted volume)
CSV_FILE_PATH = ["./train.csv", "./val.csv"]

# Mounted-volume directory where the source .flac files are stored
FLAC_INPUT_DIR = "/Volumes/gen_audio_catalog/volumes/kinh/datasets/AudioSet/full/audio/unbal_train/"  # e.g. <FLAC_INPUT_DIR>/<file_id>.flac

# Root folder for processed .wav files (train/ and val/ sub-folders are created automatically)
WAV_OUTPUT_DIR = "/Volumes/gen_audio_catalog/volumes/ziyi/Diffusion_AudioSet/processed_wav/"
TRAIN_WAV_DIR = os.path.join(WAV_OUTPUT_DIR, "train")
VAL_WAV_DIR = os.path.join(WAV_OUTPUT_DIR, "val")

# Paths for the separate training and validation JSON metadata files
JSON_OUTPUT_DIR = "/Volumes/gen_audio_catalog/volumes/ziyi/Diffusion_AudioSet/metadata/"
TRAIN_JSON_PATH = os.path.join(JSON_OUTPUT_DIR, "audiocaps_train.json")
VAL_JSON_PATH = os.path.join(JSON_OUTPUT_DIR, "audiocaps_val.json")

# Target audio parameters
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1  # mono
# COMMAND ----------
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_csv_metadata(csv_path: str):
    """Read the CSV and yield (file_id, caption) for each valid row.

    - Uses the first column as the file ID (audio file name).
    - Skips rows where the file ID starts with '='.
    - Uses the fourth column as the caption.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row
        if header:
            logger.info("CSV header: %s", header)

        for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
            if len(row) < 4:
                logger.warning("Row %d: fewer than 4 columns, skipping: %s", row_num, row)
                continue

            file_id = row[0].strip()

            # Skip rows where the file ID starts with '='
            if file_id.startswith("="):
                logger.info("Row %d: file ID starts with '=', skipping: %s", row_num, file_id)
                continue

            caption = row[3].strip()
            yield file_id, caption


def safe_write_bytes(dest_path: str, data: bytes):
    """Write *data* to *dest_path* via a local temp file, then copy.

    Mounted S3 / FUSE volumes may not support all write modes reliably.
    Writing to a local temp file first and then copying avoids partial-write
    or permission issues on such mounts.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=os.path.splitext(dest_path)[1],
        dir=tempfile.gettempdir(),
    )
    try:
        with os.fdopen(fd, "wb") as tmp_f:
            tmp_f.write(data)
        shutil.copy2(tmp_path, dest_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def safe_write_text(dest_path: str, text: str, encoding: str = "utf-8"):
    """Write *text* to *dest_path* via a local temp file, then copy."""
    safe_write_bytes(dest_path, text.encode(encoding))


def read_flac_file(flac_path: str) -> bytes:
    """Read a .flac file from the local/mounted filesystem and return its raw bytes."""
    with open(flac_path, "rb") as f:
        return f.read()


def convert_to_mono_wav(audio_bytes: bytes, target_sr: int) -> tuple:
    """Convert raw audio bytes (any format supported by soundfile) to mono at *target_sr*.

    Returns (audio_np_array, sample_rate).
    """
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)

    # Convert to mono by averaging channels
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data[:, 0]

    # Resample if necessary
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError(
                f"librosa is required for resampling ({sr} -> {target_sr}). "
                "Install it with: pip install librosa"
            )

    return data, target_sr


def save_wav(audio_data: np.ndarray, sample_rate: int, output_path: str):
    """Save a numpy audio array as a WAV file.

    Writes to an in-memory buffer first, then safely copies to *output_path*
    via a local temp file.  This avoids issues where ``soundfile.write()``
    cannot write directly to mounted-volume / FUSE paths backed by S3.
    """
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV", endian="LITTLE", subtype="PCM_16")
    safe_write_bytes(output_path, buf.getvalue())
    buf.close()


def build_json_entry(wav_path: str, caption: str) -> dict:
    """Build a single JSON entry matching the downstream expected format.

    Placeholder values are used for 'seg_label' and 'labels' since they are
    not available from the CSV metadata.
    """
    return {
        "wav": wav_path,
        "seg_label": "",
        "labels": "",
        "caption": caption,
    }

# COMMAND ----------
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _process_entries(entries, wav_dir, json_path, label=""):
    """Process a list of (file_id, caption) entries: convert .flac → .wav and
    write a JSON metadata file.

    Returns (processed_count, error_count).
    """
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    json_entries = []
    processed_count = 0
    error_count = 0
    skipped_count = 0

    for file_id, caption in tqdm(entries, desc=label, unit="file"):
        flac_path = os.path.join(FLAC_INPUT_DIR, f"{file_id}.flac")

        if not os.path.exists(flac_path):
            logger.warning("[%s] FLAC file not found, skipping: %s", label, flac_path)
            skipped_count += 1
            continue

        wav_filename = f"{file_id}.wav"
        wav_output_path = os.path.join(wav_dir, wav_filename)

        try:
            audio_bytes = read_flac_file(flac_path)

            audio_data, sr = convert_to_mono_wav(audio_bytes, TARGET_SAMPLE_RATE)

            save_wav(audio_data, sr, wav_output_path)

            json_entries.append(build_json_entry(wav_output_path, caption))
            processed_count += 1

        except Exception:
            logger.exception("[%s] Failed to process file_id=%s", label, file_id)
            error_count += 1
            continue

    output_json = {"data": json_entries}
    safe_write_text(json_path, json.dumps(output_json, indent=4, ensure_ascii=False))

    logger.info(
        "[%s] Done. Processed: %d | Skipped (missing): %d | Errors: %d | JSON: %s",
        label, processed_count, skipped_count, error_count, json_path,
    )
    return processed_count, error_count


def main(merge_val_portion: float = 0.0, seed: int = 42):
    """Prepare audio data in three ordered steps.

    Parameters
    ----------
    merge_val_portion : float
        Fraction of val.csv entries (0.0–1.0) to move into the training set.
        0.0 means no merging — all val data stays as validation.
    seed : int
        Random seed used when selecting val entries to merge.

    Workflow
    -------
    1. Process **all** training entries from train.csv → train/ folder + train JSON.
    2. If *merge_val_portion* > 0, randomly select that portion from val.csv,
       process them as .wav into the train/ folder, and **append** their
       entries to the training JSON.
    3. Process the **remaining** val entries → val/ folder + val JSON.
    """
    import random

    logger.info("Starting data preparation...")

    # ── Read CSV metadata ──────────────────────────────────────────────
    train_entries = list(read_csv_metadata("./train.csv"))
    val_entries = list(read_csv_metadata("./val.csv"))
    logger.info("CSV entries — train: %d | val: %d", len(train_entries), len(val_entries))

    # ── Split val entries into "merge-to-train" and "keep-as-val" ──────
    val_to_train = []
    val_remaining = val_entries  # default: keep all as val

    if 0.0 < merge_val_portion <= 1.0:
        random.seed(seed)
        shuffled = val_entries[:]
        random.shuffle(shuffled)
        n_move = max(1, int(len(shuffled) * merge_val_portion))
        val_to_train = shuffled[:n_move]
        remaining_ids = {fid for fid, _ in shuffled[n_move:]}
        # preserve original order for remaining entries
        val_remaining = [(fid, cap) for fid, cap in val_entries if fid in remaining_ids]
        logger.info(
            "Val split — moving to train: %d (%.1f%%) | keeping as val: %d (seed=%d)",
            len(val_to_train), merge_val_portion * 100, len(val_remaining), seed,
        )

    # ── Step 1: Process training data ──────────────────────────────────
    logger.info("=== Step 1/3: Processing TRAINING data ===")
    train_proc, train_err = _process_entries(
        train_entries, TRAIN_WAV_DIR, TRAIN_JSON_PATH, label="TRAIN",
    )

    # ── Step 2: Merge selected val entries into training ───────────────
    if val_to_train:
        logger.info("=== Step 2/3: Merging selected VAL entries into TRAINING ===")
        merge_proc, merge_err = _process_entries(
            val_to_train, TRAIN_WAV_DIR, "_tmp_merge_.json", label="VAL→TRAIN",
        )
        # Append the newly processed entries into the existing training JSON
        with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open("_tmp_merge_.json", "r", encoding="utf-8") as f:
            merge_data = json.load(f)
        train_data["data"].extend(merge_data["data"])
        safe_write_text(TRAIN_JSON_PATH, json.dumps(train_data, indent=4, ensure_ascii=False))
        # clean up temp file
        if os.path.exists("_tmp_merge_.json"):
            os.remove("_tmp_merge_.json")
        logger.info(
            "Training JSON updated — total entries: %d (original: %d + merged: %d)",
            len(train_data["data"]), train_proc, merge_proc,
        )
    else:
        logger.info("=== Step 2/3: No val entries to merge (portion=0) — skipping ===")

    # ── Step 3: Process remaining validation data ──────────────────────
    logger.info("=== Step 3/3: Processing remaining VALIDATION data ===")
    val_proc, val_err = _process_entries(
        val_remaining, VAL_WAV_DIR, VAL_JSON_PATH, label="VAL",
    )

    logger.info(
        "All done. Train: %d processed (+%d merged from val) | Val: %d processed",
        train_proc, len(val_to_train), val_proc,
    )

# COMMAND ----------
def debug(max_files: int = 10):
    """Run a quick sanity check using only *max_files* entries from val.csv."""
    logger.info("Starting DEBUG run (max %d files from val.csv)...", max_files)

    os.makedirs(VAL_WAV_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VAL_JSON_PATH), exist_ok=True)

    json_entries = []
    processed_count = 0
    error_count = 0

    entries = []
    for file_id, caption in read_csv_metadata("./val.csv"):
        entries.append((file_id, caption))
        if len(entries) >= max_files:
            break

    for file_id, caption in tqdm(entries, desc="DEBUG", unit="file"):
        flac_path = os.path.join(FLAC_INPUT_DIR, f"{file_id}.flac")

        if not os.path.exists(flac_path):
            logger.warning("[DEBUG] FLAC file not found, skipping: %s", flac_path)
            error_count += 1
            continue

        wav_filename = f"{file_id}.wav"
        wav_output_path = os.path.join(VAL_WAV_DIR, wav_filename)

        try:
            audio_bytes = read_flac_file(flac_path)

            audio_data, sr = convert_to_mono_wav(audio_bytes, TARGET_SAMPLE_RATE)

            save_wav(audio_data, sr, wav_output_path)

            json_entries.append(build_json_entry(wav_output_path, caption))
            processed_count += 1

        except Exception:
            logger.exception("[DEBUG] Failed to process file_id=%s", file_id)
            error_count += 1
            continue

    # Write a debug JSON file
    debug_json_path = VAL_JSON_PATH.replace(".json", "_debug.json")
    output_json = {"data": json_entries}
    safe_write_text(debug_json_path, json.dumps(output_json, indent=4, ensure_ascii=False))

    logger.info(
        "DEBUG done. Processed: %d | Errors: %d | JSON saved to: %s",
        processed_count,
        error_count,
        debug_json_path,
    )
# COMMAND ----------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare audio data for AudioLDM training.")
    parser.add_argument("--debug", action="store_true",default=True, help="Run a quick debug pass on 10 files from val.csv")
    parser.add_argument(
        "--merge-val",
        type=float,
        default=0.97,
        metavar="PORTION",
        help="Move PORTION (0.0–1.0) of val.csv entries into the training set. "
             "The moved files are saved as .wav in the train/ folder and added to "
             "the training JSON; they are excluded from the val set. "
             "0.0 (default) means no merging. E.g. --merge-val 0.5 moves 50%%.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --merge-val (default: 42)")
    args = parser.parse_args()

    if args.debug:
        debug()
    else:
        main(merge_val_portion=args.merge_val, seed=args.seed)

