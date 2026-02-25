# Databricks notebook source
# COMMAND ----------
# %pip install -r /Workspace/Users/ziyi.xu@harman.com/AUdioLDM2/audioldm_train/modules/audiomae/sequence_gen/requirements.txt
#
# dbutils.library.restartPython()
# COMMAND ----------
import subprocess
import sys
import os
import argparse

# COMMAND ----------
def run_diffusion(dry_run: bool) -> None:
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is not defined in some environments (e.g. Databricks notebooks)
        script_dir = os.getcwd()
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(repo_root)

    # Path to the latent diffusion training script
    main_script = os.path.join(repo_root, "audioldm_train", "train", "latent_diffusion.py")
    config_path = os.path.join(
        repo_root,
        "audioldm_train",
        "config",
        "2023_08_23_reproduce_audioldm",
        "audioldm_crossattn_Ziyi.yaml",
    )

    # Command arguments (equivalent to the terminal command in the request)
    cmd = [
        sys.executable,  # Python interpreter
        main_script,
        "-c",
        config_path,
    ]

    print("Working directory:", os.getcwd())
    print("Running AudioLDM2 with command:")
    print(" ".join(cmd))

    if dry_run:
        print("Dry run enabled. File checks:")
        print("- main_script exists:", os.path.exists(main_script))
        print("- config_path exists:", os.path.exists(config_path))
        return

    # Optional: set environment variables for verbose logging
    os.environ["AUDIO_LDM_DEBUG"] = "1"  # You can check this flag inside your code for extra logs

    # Run the process and stream output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace")

    for line in process.stdout:
        print(line.strip())

    process.wait()
    print(f"Process finished with exit code {process.returncode}")

# COMMAND ----------
def _is_databricks() -> bool:
    """Detect if running inside a Databricks notebook."""
    return (
        "DATABRICKS_RUNTIME_VERSION" in os.environ
        or "db_ipykernel_launcher" in sys.argv[0]
        or any("/databricks/" in arg or "\\databricks\\" in arg for arg in sys.argv)
        or "ipykernel_launcher" in sys.argv[0]
    )

if _is_databricks():
    # In Databricks notebooks argparse will choke on the kernel arguments,
    # so we skip argument parsing and use sensible defaults.
    run_diffusion(dry_run=False)
elif __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only validate paths without starting training.")
    args, _ = parser.parse_known_args()
    run_diffusion(dry_run=args.dry_run)
