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

import csv
import json
import os
import io
import logging

import soundfile as sf
import numpy as np

# ---------------------------------------------------------------------------
# Configuration – adjust these to match your environment
# ---------------------------------------------------------------------------
# change the base directory to the current working directory of the SCRIPT TO ENSURE THE CSV FILES ARE FOUND
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Path to the input CSV file (on DBFS or mounted volume)
CSV_FILE_PATH = ["./train.csv", "./val.csv"]

# Mounted-volume directory where the source .flac files are stored
FLAC_INPUT_DIR = "/dbfs/mnt/my-volume/audiocaps/flac/"  # e.g. <FLAC_INPUT_DIR>/<file_id>.flac

# Local (mounted-volume) folder where processed .wav files will be saved
WAV_OUTPUT_DIR = "/dbfs/mnt/my-volume/processed_wav/"

# Path (including filename) for the output JSON file
JSON_OUTPUT_PATH = "/dbfs/mnt/my-volume/metadata/audiocaps_processed.json"

# Target audio parameters
TARGET_SAMPLE_RATE = 48000
TARGET_CHANNELS = 1  # mono

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
    """Save a numpy audio array as a WAV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_data, sample_rate, subtype="PCM_16")


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("Starting data preparation...")

    # Ensure output directories exist
    os.makedirs(WAV_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)

    json_entries = []
    processed_count = 0
    error_count = 0

    for csv_path in CSV_FILE_PATH:
        for file_id, caption in read_csv_metadata(csv_path):
            flac_path = os.path.join(FLAC_INPUT_DIR, f"{file_id}.flac")
            wav_filename = f"{file_id}.wav"
            wav_output_path = os.path.join(WAV_OUTPUT_DIR, wav_filename)

            try:
                # 1. Read .flac from mounted volume
                logger.info("Reading %s ...", flac_path)
                audio_bytes = read_flac_file(flac_path)

                # 2. Convert to mono WAV at target sample rate
                audio_data, sr = convert_to_mono_wav(audio_bytes, TARGET_SAMPLE_RATE)

                # 3. Save processed .wav
                save_wav(audio_data, sr, wav_output_path)
                logger.info("Saved: %s", wav_output_path)

                # 4. Track entry for JSON
                json_entries.append(build_json_entry(wav_output_path, caption))
                processed_count += 1

            except Exception:
                logger.exception("Failed to process file_id=%s", file_id)
                error_count += 1
                continue

    # 5. Write aggregated JSON
    output_json = {"data": json_entries}

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as jf:
        json.dump(output_json, jf, indent=4, ensure_ascii=False)

    logger.info(
        "Done. Processed: %d | Errors: %d | JSON saved to: %s",
        processed_count,
        error_count,
        JSON_OUTPUT_PATH,
    )


def debug(max_files: int = 10):
    """Run a quick sanity check using only *max_files* entries from val.csv."""
    logger.info("Starting DEBUG run (max %d files from val.csv)...", max_files)

    os.makedirs(WAV_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)

    json_entries = []
    processed_count = 0
    error_count = 0

    for file_id, caption in read_csv_metadata("./val.csv"):
        if processed_count + error_count >= max_files:
            break

        flac_path = os.path.join(FLAC_INPUT_DIR, f"{file_id}.flac")
        wav_filename = f"{file_id}.wav"
        wav_output_path = os.path.join(WAV_OUTPUT_DIR, wav_filename)

        try:
            logger.info("[DEBUG %d/%d] Reading %s ...", processed_count + error_count + 1, max_files, flac_path)
            audio_bytes = read_flac_file(flac_path)

            audio_data, sr = convert_to_mono_wav(audio_bytes, TARGET_SAMPLE_RATE)

            save_wav(audio_data, sr, wav_output_path)
            logger.info("[DEBUG] Saved: %s", wav_output_path)

            json_entries.append(build_json_entry(wav_output_path, caption))
            processed_count += 1

        except Exception:
            logger.exception("[DEBUG] Failed to process file_id=%s", file_id)
            error_count += 1
            continue

    # Write a debug JSON file
    debug_json_path = JSON_OUTPUT_PATH.replace(".json", "_debug.json")
    output_json = {"data": json_entries}
    with open(debug_json_path, "w", encoding="utf-8") as jf:
        json.dump(output_json, jf, indent=4, ensure_ascii=False)

    logger.info(
        "DEBUG done. Processed: %d | Errors: %d | JSON saved to: %s",
        processed_count,
        error_count,
        debug_json_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare audio data for AudioLDM training.")
    parser.add_argument("--debug", action="store_true", help="Run a quick debug pass on 10 files from val.csv")
    args = parser.parse_args()

    if args.debug:
        debug()
    else:
        main()

