import subprocess
import sys
import os
import argparse

def run_diffusion(dry_run: bool) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(repo_root)

    # Path to the latent diffusion training script
    main_script = os.path.join(repo_root, "audioldm_train", "train", "latent_diffusion.py")
    config_path = os.path.join(
        repo_root,
        "audioldm_train",
        "config",
        "2023_08_23_reproduce_audioldm",
        "audioldm_crossattn_T5_Clap_Ziyi.yaml",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only validate paths without starting training.")
    args = parser.parse_args()

    run_diffusion(dry_run=args.dry_run)
