import torch
import soundfile as sf
from omegaconf import OmegaConf
import sys

import os
import sys

# Absolute path to your project root (where `data/` lives)
project_root = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning".replace("\\", "/")

# Change working directory
os.chdir(project_root)

# Optional: Add project root to sys.path for imports
sys.path.append(project_root)

print("Current working directory:", os.getcwd())

import torch
import soundfile as sf
from omegaconf import OmegaConf
from audioldm_train.utilities.model_util import instantiate_from_config
import numpy as np

import os
print("Current working directory:", os.getcwd())



cfg_path  = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\audioldm_train\config\2023_08_23_reproduce_audioldm\audioldm2_full.yaml".replace("\\", "/")
ckpt_path = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\data\checkpoints\vae_mel_16k_64bins.ckpt".replace("\\", "/")
MAE_emb_path = r"C:\Users\ZiXu\Downloads\gpt2_output\0000_20251104_171136_mae_embeds.npy".replace("\\", "/")
MAE_attn_path = r"C:\Users\ZiXu\Downloads\gpt2_output\0000_20251104_171136_mae_attn_mask.npy".replace("\\", "/")

# --- paths you must fill ---
save_path  = r"C:\Users\ZiXu\Downloads\gpt2_output\generated_ddpm_audiomae.wav".replace("\\", "/")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Load Model
# -----------------------------
print("Loading model...")
config = OmegaConf.load(cfg_path)
model = instantiate_from_config(config["model"])

state = torch.load(ckpt_path, map_location="cpu")
state_dict = state["state_dict"] if "state_dict" in state else state
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys")

model = model.to(device).eval()
model.conditional_dry_run_finished = True  # Disable unconditional CFG during inference

# -----------------------------
# 3. Prepare AudioMAE Conditions
# -----------------------------
# Assume you already have AudioMAE outputs:
# audio_mae_tokens: Tensor [B, seq_len, dim]
# attention_mask: Tensor [B, seq_len] or similar
# Replace these with your actual tensors:
# audio_mae_tokens = torch.randn(1, 128, 768).to(device)  # Example dummy tensor
# attention_mask = torch.ones(1, 128).to(device)          # Example dummy mask
# Load from .npy files
audio_mae_tokens = torch.from_numpy(np.load(MAE_emb_path)).to(device)
attention_mask = torch.from_numpy(np.load(MAE_attn_path)).to(device)

cond = {
    "film_clap_cond1": audio_mae_tokens
}

batch_size = audio_mae_tokens.size(0)

# -----------------------------
# 4. DDPM Sampling
# -----------------------------
timesteps = 1000  # Full DDPM steps for best quality
shape = (model.channels, model.latent_t_size, model.latent_f_size)

print("Starting DDPM sampling...")
with model.ema_scope("Inference"):  # Use EMA weights for better quality
    samples = model.p_sample_loop(
        cond=cond,
        shape=(batch_size, *shape),
        timesteps=timesteps,
        verbose=True
    )

# -----------------------------
# 5. Decode to Waveform
# -----------------------------
print("Decoding mel spectrogram to waveform...")
mel = model.decode_first_stage(samples)  # [B, 1, T, F]
waveform = model.mel_spectrogram_to_waveform(mel, save=False)

# Normalize and save
wav = waveform[0, 0]
wav = (wav / (abs(wav).max() + 1e-8)) * 0.99
sf.write(save_path, wav, samplerate=model.sampling_rate)
print(f"Audio saved to: {save_path}")


# import torch
# import os
# import sys
#
# # Absolute path to your project root (where `data/` lives)
# project_root = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning".replace("\\", "/")
#
# # 1) Load your trained LatentDiffusion checkpoint
# #    (make sure the same config is used as in training)
# from audioldm_train.utilities.model_util import instantiate_from_config
# from omegaconf import OmegaConf
#
# cfg_path  = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\audioldm_train\config\2023_08_23_reproduce_audioldm\audioldm2_full.yaml".replace("\\", "/")
# ckpt_path = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\data\checkpoints\vae_mel_16k_64bins.ckpt".replace("\\", "/")
# MAE_emb_path = r"C:\Users\ZiXu\Downloads\gpt2_output\0000_20251104_171136_clap_emb.npy".replace("\\", "/")
# MAE_attn_path = r"C:\Users\ZiXu\Downloads\gpt2_output\0000_20251104_171136_mae_attn_mask.npy".replace("\\", "/")
#
# cfg = OmegaConf.load("path/to/your/config.yaml")  # the config used to train this model
# ldm = instantiate_from_config(cfg.model)
# ckpt = torch.load("path/to/your/ldm.ckpt", map_location="cpu")
# missing, unexpected = ldm.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
# print("Missing:", len(missing), "Unexpected:", len(unexpected))
#
# ldm.eval().cuda()
#
# # 2) Prepare AudioMAE context
# #    Suppose you already have AudioMAE outputs: `audiomae_tokens` of shape [B, N, 768]
# #    and (optionally) a binary mask `audiomae_mask` of shape [B, N]
# audiomae_tokens = your_audiomae_tokens.float().cuda()      # [B, N, D]
# if "audiomae_mask" in locals() and audiomae_mask is not None:
#     attn_mask = audiomae_mask.bool().cuda()                # [B, N]
# else:
#     attn_mask = torch.ones(audiomae_tokens.size()[:2], dtype=torch.bool, device=audiomae_tokens.device)
#
# B = audiomae_tokens.size(0)
#
# # 3) Build the cond dict with the EXACT key used during training
# #    Check what your model expects:
# print("conditioning_key list:", ldm.conditioning_key)
# print("metadata keys:      ", list(ldm.cond_stage_model_metadata.keys()))
# # Replace the key below with one of the keys above (e.g. "crossattn_audiomae")
# cond_key = "crossattn_audiomae"
#
# cond = {
#     cond_key: (audiomae_tokens, attn_mask)   # tuple is accepted by DiffusionWrapper
# }
#
# # Optional: Classifier-free guidance (CFG)
# # If you want CFG > 1.0 you must supply an "unconditional_conditioning"
# # with the same structure:
# use_cfg = False
# cfg_scale = 1.0
# uncond = None
# if use_cfg and cfg_scale != 1.0:
#     uncond = {cond_key: (torch.zeros_like(audiomae_tokens), attn_mask)}
#
# # 4) Sample latent mels with DDIM or PLMS
# ddim_steps = 200
# use_plms = False  # True if you prefer PLMS
# with ldm.ema_scope("infer"):
#     samples, _ = ldm.sample_log(
#         cond=cond,
#         batch_size=B,
#         ddim=True,
#         ddim_steps=ddim_steps,
#         unconditional_guidance_scale=cfg_scale,
#         unconditional_conditioning=uncond,
#         use_plms=use_plms
#     )
#
# # 5) Decode latent to mel, then to waveform and save
# mel = ldm.decode_first_stage(samples)  # -> [B, 1, T, F]
# # Save files to disk (names auto-generated) and return numpy waveform
# wave = ldm.mel_spectrogram_to_waveform(mel, savepath="./gen", name="from_audiomae", save=True)
# print("Saved to ./gen/*.wav")