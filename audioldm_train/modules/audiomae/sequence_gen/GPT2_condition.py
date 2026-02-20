"""Helper for generating GPT-2 conditioning embeddings from raw text.

This module mirrors the inference behavior in ``inference_CLAP_T5_GPT2.py`` but exposes
an in-memory function for use inside training (e.g., diffusion conditioning). It loads
CLAP (text) + FLAN-T5 encoders, restores the Lightning-wrapped GPT-2 adapter weights,
and returns the normalized GPT-2 predictions on the caller's device.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import GPT2Model

from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2, FlanT5HiddenState


_DEFAULT_GPT2_CKPT = (
    "C:/Users/ZiXu/Documents/Python_Scripts/AudioLDM_GPT2_Checkpoint/checkpoints/"
    "clapT52gpt2/20260202_140247/GPT2_epoch=10_val_loss=0.0291.ckpt"
)


class CLAPT5ToGPT2(nn.Module):
    def __init__(
        self,
        sequence_input_key,
        sequence_input_embed_dim,
        target_embed_dim: int = 768,
        max_seq_len: int = 1024,
        mae_token_num: int = 0,
        gpt2_name: str = "gpt2",
    ):
        super().__init__()
        assert len(sequence_input_key) == len(sequence_input_embed_dim)
        self.sequence_input_key = sequence_input_key
        self.sequence_input_embed_dim = sequence_input_embed_dim
        self.target_tokens_mask_ratio = 0.0
        self.target_embed_dim = target_embed_dim
        self.max_seq_len = max_seq_len
        self.mae_token_num = mae_token_num
        self.start_of_sequence_tokens = nn.Embedding(32, target_embed_dim)
        self.end_of_sequence_tokens = nn.Embedding(32, target_embed_dim)
        self.input_sequence_embed_linear = nn.ModuleList(
            [nn.Linear(dim, target_embed_dim) for dim in sequence_input_embed_dim]
        )
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        batch = sequence.size(0)
        new_attn_mask_step = torch.ones((batch, 1), device=sequence.device)
        new_attn_mask = torch.cat([new_attn_mask_step, attn_mask, new_attn_mask_step], dim=1)
        key_id = torch.tensor([_id], device=sequence.device)
        sos_token = self.start_of_sequence_tokens(key_id).expand(batch, 1, -1)
        eos_token = self.end_of_sequence_tokens(key_id).expand(batch, 1, -1)
        new_sequence = torch.cat([sos_token, sequence, eos_token], dim=1)
        return new_sequence, new_attn_mask

    def truncate_sequence_and_mask(self, sequence, mask, max_len=512):
        if sequence.size(1) > max_len:
            return sequence[:, :max_len], mask[:, :max_len]
        return sequence, mask

    def get_input_sequence_and_mask(self, cond_dict):
        input_embeds, input_mask = None, None
        for _id, key in enumerate(self.sequence_input_key):
            assert key in cond_dict, f"Missing key {key}"
            cond_embed = cond_dict[key]
            if isinstance(cond_embed, list):
                seq, mask = cond_embed
            else:
                seq = cond_embed
                if seq.dim() == 2:
                    seq = seq.unsqueeze(1)
                mask = torch.ones((seq.size(0), seq.size(1)), device=seq.device)
            seq = self.input_sequence_embed_linear[_id](seq)
            seq, mask = self.add_sos_eos_tokens(_id, seq, mask)
            if input_embeds is None:
                input_embeds, input_mask = seq, mask
            else:
                input_embeds = torch.cat([input_embeds, seq], dim=1)
                input_mask = torch.cat([input_mask, mask], dim=1)
        max_len = int(self.max_seq_len - self.mae_token_num)
        input_embeds, input_mask = self.truncate_sequence_and_mask(input_embeds, input_mask, max_len)
        cond_end_idx = input_embeds.size(1)
        return input_embeds, input_mask, cond_end_idx

    def forward(self, cond_dict, target_embeds, target_mask):
        input_embeds, input_mask, cond_end_idx = self.get_input_sequence_and_mask(cond_dict)
        final_input_embeds = torch.cat([input_embeds, target_embeds], dim=1)
        final_input_mask = torch.cat([input_mask, target_mask], dim=1)
        output_embeds = self.gpt2(inputs_embeds=final_input_embeds, attention_mask=final_input_mask.long())["last_hidden_state"]
        output = output_embeds[:, cond_end_idx - 1 : -1, :]
        return output


class LitCLAPT5ToGPT2(pl.LightningModule):
    def __init__(
        self,
        clap_model,
        T5_model,
        sequence_input_key,
        sequence_input_embed_dim,
        target_embed_dim: int = 768,
        max_seq_len: int = 1024,
        mae_token_num: int = 0,
        gpt2_name: str = "gpt2",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["clap_model", "T5_model"])
        self.clap = clap_model.eval()
        for param in self.clap.parameters():
            param.requires_grad = False
        self.T5 = T5_model.eval()
        for param in self.T5.parameters():
            param.requires_grad = False
        self.model = CLAPT5ToGPT2(
            sequence_input_key=sequence_input_key,
            sequence_input_embed_dim=sequence_input_embed_dim,
            target_embed_dim=target_embed_dim,
            max_seq_len=max_seq_len,
            mae_token_num=mae_token_num,
            gpt2_name=gpt2_name,
        )

    def forward(self, cond_dict, target_embeds, target_mask):
        return self.model(cond_dict, target_embeds, target_mask)

    @torch.no_grad()
    def generate(self, cond_dict: dict, num_tokens: int, device: torch.device):
        batch = None
        for value in cond_dict.values():
            seq = value[0] if isinstance(value, list) else value
            batch = seq.size(0)
            break
        hidden_dim = self.model.target_embed_dim
        prefix = torch.zeros((batch, 0, hidden_dim), device=device, dtype=torch.float32)
        for _ in range(num_tokens):
            if prefix.size(1) == 0:
                target_in = torch.zeros((batch, 1, hidden_dim), device=device, dtype=torch.float32)
                mask_in = torch.ones((batch, 1), device=device, dtype=torch.float32)
                pred = self(cond_dict, target_in, mask_in)
                next_token = pred[:, -1, :]
                prefix = torch.zeros((batch, 0, hidden_dim), device=device, dtype=torch.float32)
            else:
                mask_in = torch.ones((batch, prefix.size(1)), device=device, dtype=torch.float32)
                pred = self(cond_dict, prefix, mask_in)
                next_token = pred[:, -1, :]
            prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
        return prefix


_ModelCache = {}


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_clap_checkpoint(user_path: Optional[str]) -> Path:
    if user_path:
        return Path(user_path)
    return _resolve_repo_root() / "data" / "checkpoints" / "clap_htsat_tiny.pt"


def _load_models(
    checkpoint_path: str,
    device: torch.device,
    max_seq_len: int,
    gpt2_name: str,
    clap_checkpoint_path: Optional[str],
) -> Tuple[LitCLAPT5ToGPT2, CLAPAudioEmbeddingClassifierFreev2, FlanT5HiddenState]:
    key = (str(device), Path(checkpoint_path).resolve(), max_seq_len, gpt2_name, clap_checkpoint_path)
    if key in _ModelCache:
        return _ModelCache[key]

    clap_ckpt = _resolve_clap_checkpoint(clap_checkpoint_path)
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not clap_ckpt.is_file():
        raise FileNotFoundError(f"CLAP checkpoint not found: {clap_ckpt}")

    clap = CLAPAudioEmbeddingClassifierFreev2(
        pretrained_path=str(clap_ckpt),
        embed_mode="text",
        amodel="HTSAT-tiny",
        unconditional_prob=0.0,
        training_mode=False,
    ).to(device)
    clap.eval()

    t5 = FlanT5HiddenState(text_encoder_name="google/flan-t5-large", freeze_text_encoder=True).to(device)
    t5.eval()

    lit_model = LitCLAPT5ToGPT2(
        clap_model=clap,
        T5_model=t5,
        sequence_input_key=["Clap_text_encoder", "T5_text_encoder"],
        sequence_input_embed_dim=[512, 1024],
        target_embed_dim=768,
        max_seq_len=max_seq_len,
        mae_token_num=0,
        gpt2_name=gpt2_name,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    inner_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    lit_model.model.load_state_dict(inner_state, strict=False)
    lit_model.eval()

    _ModelCache[key] = (lit_model, clap, t5)
    return lit_model, clap, t5


@torch.no_grad()
def generate_gpt2_condition(
    texts: List[str],
    checkpoint_path: str = _DEFAULT_GPT2_CKPT,
    device: Optional[torch.device] = None,
    max_seq_len: int = 1024,
    gpt2_name: str = "gpt2",
    clap_checkpoint_path: Optional[str] = None,
) -> torch.Tensor:
    """Generate normalized GPT-2 conditioning embeddings for a batch of texts.

    Args:
        texts: List of input strings.
        checkpoint_path: Path to the Lightning checkpoint containing GPT-2 adapter weights.
            Defaults to the path used in ``inference_CLAP_T5_GPT2.py``.
        device: Torch device for all computations. Defaults to CUDA if available else CPU.
        max_seq_len: Maximum sequence length for GPT-2 inputs (matches training setup).
        gpt2_name: HF GPT-2 variant to instantiate.
        clap_checkpoint_path: Optional override for CLAP text checkpoint path.

    Returns:
        Tensor of shape [batch, sequence_len, 768] containing ``gpt2_prediction_normalized``
        on the requested device.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list of strings")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lit_model, clap, t5 = _load_models(
        checkpoint_path=checkpoint_path,
        device=device,
        max_seq_len=max_seq_len,
        gpt2_name=gpt2_name,
        clap_checkpoint_path=clap_checkpoint_path,
    )

    clap_embeddings = clap(texts)
    t5_hidden, t5_mask = t5(texts)

    cond_dict = {
        "Clap_text_encoder": clap_embeddings,
        "T5_text_encoder": [t5_hidden, t5_mask],
    }

    num_tokens = t5_hidden.shape[1]
    gpt2_pred = lit_model.generate(cond_dict=cond_dict, num_tokens=num_tokens, device=device)

    max_abs = torch.max(torch.abs(gpt2_pred)).clamp(min=1e-12)
    gpt2_norm = gpt2_pred / max_abs
    return gpt2_norm

