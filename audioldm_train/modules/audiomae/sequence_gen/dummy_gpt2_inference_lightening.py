#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug-friendly inference for CLAP -> GPT-2 next-token embedding generation.

- All settings are specified as variables below (no argparse).
- Rebuilds LitCLAPToGPT2 and CLAPToGPT2 exactly like in training.
- Loads Lightning checkpoint (CLAP was excluded in on_save_checkpoint).
- Generates S tokens of 768-dim embeddings, conditioned on CLAP text embeddings.
- Saves per-prompt .npy outputs: *_mae_embeds.npy (S,768) and *_mae_attn_mask.npy (S,).

Notes:
- Generation is deterministic by default (regressing embeddings). Optional tiny noise can be added.
- No prefix support (removed entirely as requested).
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import GPT2Model


# =========
# SETTINGS (edit here)
# =========

# Paths
CHECKPOINT_PATH = r"/Volumes/gen_audio_catalog/volumes/ziyi/GPT2_training/checkpoints/clap2gpt2/REPLACE_WITH_RUN/GPT2_epoch=XX_val_loss=YY.ckpt"
CLAP_WEIGHTS    = r"/Volumes/gen_audio_catalog/volumes/ziyi/Checkpoint_AudioLDM2/clap_htsat_tiny.pt"
OUT_DIR         = r"/Volumes/gen_audio_catalog/volumes/ziyi/GPT2_training/generated_debug"

# Prompts (batch inference supported)
PROMPTS = [
    "a calm ambient soundscape with gentle pads",
    "busy street with honking and chatter",
]

# Decoding/generation
NUM_TOKENS      = 16        # number of tokens to generate per prompt
ADD_NOISE       = False     # set True to inject tiny gaussian noise for diversity
NOISE_STD       = 0.01      # std for gaussian noise if ADD_NOISE=True

# GPT-2 backbone name (must match training if you changed it)
GPT2_NAME       = "gpt2"

# Device preference
DEVICE          = "auto"    # "cuda" | "cpu" | "auto"


# =========
# ENV IMPORTS (CLAP wrapper)
# =========
# Ensure this path is correct in your environment
sys.path.append(r"/Workspace/Users/ziyi.xu@harman.com/AUdioLDM2")
from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2


# =========================
# Model definitions (same as training)
# =========================
class CLAPToGPT2(nn.Module):
    """
    Combines conditioning streams (e.g., CLAP text) and a GPT-2 backbone that
    predicts next-token embeddings (regression on AudioMAE pooled tokens).
    """
    def __init__(self,
                 sequence_input_key,
                 sequence_input_embed_dim,
                 target_embed_dim=768,
                 max_seq_len=1024,
                 mae_token_num=0,
                 gpt2_name="gpt2"):
        super().__init__()
        assert len(sequence_input_key) == len(sequence_input_embed_dim), \
            "Keys and dims must match."

        self.sequence_input_key = sequence_input_key
        self.sequence_input_embed_dim = sequence_input_embed_dim
        self.target_embed_dim = target_embed_dim
        self.max_seq_len = max_seq_len
        self.mae_token_num = mae_token_num

        # Learned SOS/EOS embeddings for up to 32 sequence types
        self.start_of_sequence_tokens = nn.Embedding(32, target_embed_dim)
        self.end_of_sequence_tokens   = nn.Embedding(32, target_embed_dim)

        # Linear projections per conditioning stream
        self.input_sequence_embed_linear = nn.ModuleList([
            nn.Linear(dim, target_embed_dim) for dim in sequence_input_embed_dim
        ])

        # GPT-2 backbone
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        """
        sequence: [B, T, D], attn_mask: [B, T]
        """
        B = sequence.size(0)
        new_attn_mask_step = torch.ones((B, 1), device=sequence.device)
        new_attn_mask = torch.cat([new_attn_mask_step, attn_mask, new_attn_mask_step], dim=1)

        key_id = torch.tensor([_id], device=sequence.device)
        sos = self.start_of_sequence_tokens(key_id).expand(B, 1, -1)
        eos = self.end_of_sequence_tokens(key_id).expand(B, 1, -1)

        new_sequence = torch.cat([sos, sequence, eos], dim=1)
        return new_sequence, new_attn_mask

    def truncate_sequence_and_mask(self, sequence, mask, max_len=512):
        if sequence.size(1) > max_len:
            print(f"[warn] Truncating sequence from {sequence.size(1)} to {max_len}")
            return sequence[:, :max_len], mask[:, :max_len]
        return sequence, mask

    def get_input_sequence_and_mask(self, cond_dict):
        """
        cond_dict: { key: Tensor [B,T,D] or [Tensor, mask] }
        Returns: input_embeds [B,S,768], attn_mask [B,S], cond_end_idx (int)
        """
        input_embeds, input_mask = None, None

        for _id, key in enumerate(self.sequence_input_key):
            assert key in cond_dict, f"Missing key {key}"
            cond_embed = cond_dict[key]
            if isinstance(cond_embed, list):
                seq, mask = cond_embed
            else:
                seq = cond_embed
                if seq.dim() == 2:  # [B,D] -> [B,1,D]
                    seq = seq.unsqueeze(1)
                mask = torch.ones((seq.size(0), seq.size(1)), device=seq.device)

            # project to GPT-2 hidden size
            seq = self.input_sequence_embed_linear[_id](seq)
            seq, mask = self.add_sos_eos_tokens(_id, seq, mask)

            if input_embeds is None:
                input_embeds, input_mask = seq, mask
            else:
                input_embeds = torch.cat([input_embeds, seq], dim=1)
                input_mask   = torch.cat([input_mask,   mask], dim=1)

        max_len = int(self.max_seq_len - self.mae_token_num)
        input_embeds, input_mask = self.truncate_sequence_and_mask(input_embeds, input_mask, max_len)
        cond_end_idx = input_embeds.size(1)
        return input_embeds, input_mask, cond_end_idx

    def forward(self, cond_dict, target_embeds, target_mask):
        """
        Returns: predicted embeddings aligned for next-token prediction
        """
        input_embeds, input_mask, cond_end_idx = self.get_input_sequence_and_mask(cond_dict)

        final_input_embeds = torch.cat([input_embeds, target_embeds], dim=1)
        final_input_mask   = torch.cat([input_mask,   target_mask],   dim=1)

        output_embeds = self.gpt2(
            inputs_embeds=final_input_embeds,
            attention_mask=final_input_mask.long()
        )["last_hidden_state"]

        # Slice to align each position with its "next" embedding
        output = output_embeds[:, cond_end_idx - 1 : -1, :]  # [B, S_tgt, H]
        return output


class LitCLAPToGPT2(pl.LightningModule):
    def __init__(self,
                 clap_model,
                 sequence_input_key,
                 sequence_input_embed_dim,
                 target_embed_dim=768,
                 max_seq_len=1024,
                 mae_token_num=0,
                 gpt2_name="gpt2"):
        super().__init__()
        self.save_hyperparameters(ignore=["clap_model"])

        # Frozen CLAP
        self.clap = clap_model.eval()
        for p in self.clap.parameters():
            p.requires_grad = False

        self.model = CLAPToGPT2(
            sequence_input_key=sequence_input_key,
            sequence_input_embed_dim=sequence_input_embed_dim,
            target_embed_dim=target_embed_dim,
            max_seq_len=max_seq_len,
            mae_token_num=mae_token_num,
            gpt2_name=gpt2_name,
        )

        self.loss_fn = nn.L1Loss()  # not used in inference, retained for completeness

    def forward(self, cond_dict, target_embeds, target_mask):
        return self.model(cond_dict, target_embeds, target_mask)


# =========================
# Inference helpers
# =========================
@torch.no_grad()
def build_clap(pretrained_path: str, device: torch.device):
    model_clap = CLAPAudioEmbeddingClassifierFreev2(
        pretrained_path=pretrained_path,
        embed_mode="text",
        amodel="HTSAT-tiny",
        unconditional_prob=0.0,
        training_mode=False,
    ).to(device)
    model_clap.eval()
    return model_clap


@torch.no_grad()
def load_lit_from_ckpt(ckpt_path: str, clap_path: str, device: torch.device) -> LitCLAPToGPT2:
    """
    Rebuild LitCLAPToGPT2 with hparams from ckpt and load the trained weights.
    """
    # Recreate CLAP (not stored in ckpt)
    clap = build_clap(clap_path, device)

    ckpt = torch.load(ckpt_path, map_location=device)
    hparams = ckpt.get("hyper_parameters", {})

    # Fallbacks mirror training defaults
    sequence_input_key        = hparams.get("sequence_input_key", ["Clap_text_encoder"])
    sequence_input_embed_dim  = hparams.get("sequence_input_embed_dim", [512])
    target_embed_dim          = hparams.get("target_embed_dim", 768)
    max_seq_len               = hparams.get("max_seq_len", 1024)
    mae_token_num             = hparams.get("mae_token_num", 0)
    gpt2_name                 = hparams.get("gpt2_name", GPT2_NAME)

    model = LitCLAPToGPT2(
        clap_model=clap,
        sequence_input_key=sequence_input_key,
        sequence_input_embed_dim=sequence_input_embed_dim,
        target_embed_dim=target_embed_dim,
        max_seq_len=max_seq_len,
        mae_token_num=mae_token_num,
        gpt2_name=gpt2_name,
    ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[warn] Missing keys when loading: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys when loading: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    return model


@torch.no_grad()
def encode_prompts_with_clap(model_lit: LitCLAPToGPT2, prompts: list[str], device: torch.device):
    """
    Encode a list of text prompts into CLAP embeddings shaped [B, 1, 512].
    """
    clap_emb = model_lit.clap(prompts).to(device)  # [B, 1, 512]
    return clap_emb


@torch.no_grad()
def generate_embeddings(model_lit: LitCLAPToGPT2,
                        clap_emb: torch.Tensor,
                        num_tokens: int,
                        device: torch.device,
                        add_noise: bool = False,
                        noise_std: float = 0.01):
    """
    Autoregressive generation of AudioMAE-pooled token embeddings (no prefix).

    Args:
        model_lit: LitCLAPToGPT2 in eval mode
        clap_emb: [B, 1, 512]
        num_tokens: number of tokens (S) to generate
        add_noise: add small gaussian noise to each generated step
        noise_std: std of noise

    Returns:
        gen_embeds: [B, S, 768] float32
        gen_mask:   [B, S] int64 (ones)
    """
    B = clap_emb.size(0)
    H = 768  # GPT-2 small hidden size / target embed dim

    # Start with empty target prefix
    prefix = torch.zeros((B, 0, H), device=device, dtype=torch.float32)
    cond_dict = {"Clap_text_encoder": clap_emb}

    for t in range(num_tokens):
        if prefix.size(1) == 0:
            # Bootstrap: single dummy token so slicing works
            target_in = torch.zeros((B, 1, H), device=device, dtype=torch.float32)
            mask_in   = torch.ones((B, 1), device=device, dtype=torch.float32)
            pred = model_lit(cond_dict, target_in, mask_in)  # [B,1,768]
            next_token = pred[:, -1, :]
            # reset prefix to empty then append
            prefix = torch.zeros((B, 0, H), device=device, dtype=torch.float32)
        else:
            mask_in = torch.ones((B, prefix.size(1)), device=device, dtype=torch.float32)
            pred = model_lit(cond_dict, prefix, mask_in)  # [B, T_cur, 768]
            next_token = pred[:, -1, :]

        if add_noise:
            next_token = next_token + noise_std * torch.randn_like(next_token)

        prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)

    gen_embeds = prefix  # [B, S, 768]
    gen_mask   = torch.ones((B, num_tokens), device=device, dtype=torch.long)
    return gen_embeds.float(), gen_mask


def save_batch_outputs(out_dir: str, prompts: list[str], embeds: torch.Tensor, masks: torch.Tensor):
    """
    Save per-prompt .npy files: embeddings and attention masks.
    """
    os.makedirs(out_dir, exist_ok=True)
    embeds_np = embeds.detach().cpu().numpy()  # [B,S,768]
    masks_np  = masks.detach().cpu().numpy()   # [B,S]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = []
    for i, _ in enumerate(prompts):
        safe_id = f"{i:04d}_{timestamp}"
        emb_path = os.path.join(out_dir, f"{safe_id}_mae_embeds.npy")
        msk_path = os.path.join(out_dir, f"{safe_id}_mae_attn_mask.npy")
        np.save(emb_path, embeds_np[i])
        np.save(msk_path,  masks_np[i])
        paths.append((emb_path, msk_path))
    return paths


def main():
    # Device resolve
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    print(f"[info] Using device: {device}")

    # Load model from checkpoint
    print("[info] Loading model...")
    model_lit = load_lit_from_ckpt(CHECKPOINT_PATH, CLAP_WEIGHTS, device)
    model_lit.eval()

    # Encode prompts via CLAP
    print("[info] Encoding prompts with CLAP...")
    clap_emb = encode_prompts_with_clap(model_lit, PROMPTS, device)  # [B,1,512]

    # Generate
    print(f"[info] Generating {NUM_TOKENS} tokens...")
    t0 = time.time()
    embeds, masks = generate_embeddings(
        model_lit=model_lit,
        clap_emb=clap_emb,
        num_tokens=NUM_TOKENS,
        device=device,
        add_noise=ADD_NOISE,
        noise_std=NOISE_STD
    )
    dt = time.time() - t0
    print(f"[info] Generation done in {dt:.2f}s. Output shape: {tuple(embeds.shape)}")

    # Save
    saved = save_batch_outputs(OUT_DIR, PROMPTS, embeds, masks)
    print("[info] Saved files:")
    for emb_path, msk_path in saved:
        print(f"  - {emb_path}")
        print(f"  - {msk_path}")


if __name__ == "__main__":
    main()
