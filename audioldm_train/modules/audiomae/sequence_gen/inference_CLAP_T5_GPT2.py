"""
Inference script for LitCLAPT5ToGPT2
- Loads a Lightning checkpoint
- Reads .txt files from an input folder
- Computes CLAP and FLAN T5 embeddings
- Runs GPT-2 forward pass (aligned to T5 sequence length via autoregressive next-token generation)
- Saves output dictionary per input file as .npy (same base filename)

Assumptions:
- We align GPT-2 prediction length to the FLAN T5 hidden-state sequence length by using
  an autoregressive loop that generates one token at a time until reaching the T5 length.
- The model is conditioned on both CLAP and T5 sequences, and the target sequence grows
  by feeding back the last predicted token at each step.
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Ensure repo is on path
sys.path.append(r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning")
from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2, FlanT5HiddenState

# Import the LightningModule from the training script namespace
# The class LitCLAPT5ToGPT2 is defined in dummy_train_T5_CLAP_Lightning_example.py; we mirror its logic here.
# To avoid circular imports or dependencies on that file, we re-define only what we need to load and run.
from transformers import GPT2Model
import torch.nn as nn

class CLAPT5ToGPT2(nn.Module):
    def __init__(
        self,
        sequence_input_key,
        sequence_input_embed_dim,
        target_embed_dim=768,
        max_seq_len=1024,
        mae_token_num=0,
        gpt2_name="gpt2",
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
        self.input_sequence_embed_linear = nn.ModuleList([
            nn.Linear(dim, target_embed_dim) for dim in sequence_input_embed_dim
        ])
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        B = sequence.size(0)
        new_attn_mask_step = torch.ones((B, 1), device=sequence.device)
        new_attn_mask = torch.cat([new_attn_mask_step, attn_mask, new_attn_mask_step], dim=1)
        key_id = torch.tensor([_id], device=sequence.device)
        sos_token = self.start_of_sequence_tokens(key_id).expand(B, 1, -1)
        eos_token = self.end_of_sequence_tokens(key_id).expand(B, 1, -1)
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
        target_embed_dim=768,
        max_seq_len=1024,
        mae_token_num=0,
        gpt2_name="gpt2",
        **kwargs,
    ):
        super().__init__()
        # Ignore heavy modules in hparams
        self.save_hyperparameters(ignore=["clap_model", "T5_model"])
        self.clap = clap_model.eval()
        for p in self.clap.parameters():
            p.requires_grad = False
        self.T5 = T5_model.eval()
        for p in self.T5.parameters():
            p.requires_grad = False
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
        """
        Core autoregressive generation inside the LightningModule.
        - Consumes CLAP + T5 conditioning via cond_dict.
        - Iteratively generates GPT-2 target embeddings one token at a time.
        Returns: [B, num_tokens, target_embed_dim]
        """
        # infer batch size from any conditioning entry
        B = None
        for v in cond_dict.values():
            if isinstance(v, list):
                seq = v[0]
            else:
                seq = v
            B = seq.size(0)
            break
        H = self.model.target_embed_dim
        prefix = torch.zeros((B, 0, H), device=device, dtype=torch.float32)
        for _ in range(num_tokens):
            if prefix.size(1) == 0:
                target_in = torch.zeros((B, 1, H), device=device, dtype=torch.float32)
                mask_in = torch.ones((B, 1), device=device, dtype=torch.float32)
                pred = self(cond_dict, target_in, mask_in)
                next_token = pred[:, -1, :]
                prefix = torch.zeros((B, 0, H), device=device, dtype=torch.float32)
            else:
                mask_in = torch.ones((B, prefix.size(1)), device=device, dtype=torch.float32)
                pred = self(cond_dict, prefix, mask_in)
                next_token = pred[:, -1, :]
            prefix = torch.cat([prefix, next_token.unsqueeze(1)], dim=1)
        return prefix  # [B,num_tokens,768]


# Simple dataset to read .txt files
class TextFolderDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.txt')])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        with open(os.path.join(self.folder, fname), 'r', encoding='utf-8') as fh:
            text = fh.read().strip()
        return {"filename": fname, "text": text}

# Collate to keep batch of texts and filenames
def collate_text(batch):
    texts = [item["text"] for item in batch]
    filenames = [item["filename"] for item in batch]
    return {"texts": texts, "filenames": filenames}


# ===== Editable defaults for running without CLI =====
DEFAULT_CONFIG = {
    "input_folder": r"C:\Users\ZiXu\Documents\Python_Scripts\mae_output_new\texts",   # change to your .txt folder
    "output_folder": r"C:\Users\ZiXu\Documents\Python_Scripts\AudioLDM_GPT2_output", # change to your desired output folder
    "checkpoint": r"C:\Users\ZiXu\Documents\Python_Scripts\AudioLDM_GPT2_Checkpoint\checkpoints\clapT52gpt2\20260129_141524\GPT2_epoch=24_val_loss=0.1198.ckpt", # change to your Lightning checkpoint
    "device": "auto",            # "auto" | "cpu" | "cuda"
    "batch_size": 1,              # keep 1 for per-file inference
    "max_seq_len": 1024,
}
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Inference for LitCLAPT5ToGPT2")
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing .txt files')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to write .npy outputs')
    parser.add_argument('--checkpoint', type=str, default='./best_model.ckpt', help='Path to Lightning checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (set to 1 for per-file processing)')
    parser.add_argument('--max_seq_len', type=int, default=1024, help='Max sequence length for model')

    # If no CLI args are provided, use DEFAULT_CONFIG; else parse CLI
    if len(sys.argv) == 1:
        cfg = DEFAULT_CONFIG.copy()
        # Minimal validation
        if not os.path.isdir(cfg["input_folder"]):
            raise ValueError(f"Input folder not found: {cfg['input_folder']}")
        os.makedirs(cfg["output_folder"], exist_ok=True)
        class Args:
            input_folder = cfg["input_folder"]
            output_folder = cfg["output_folder"]
            checkpoint = cfg["checkpoint"]
            device = cfg["device"]
            batch_size = cfg["batch_size"]
            max_seq_len = cfg["max_seq_len"]
        args = Args()
    else:
        args = parser.parse_args()
        os.makedirs(args.output_folder, exist_ok=True)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Build CLAP and T5 (frozen)
    pretrained_path = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\data\checkpoints\clap_htsat_tiny.pt".replace("\\", "/")
    clap = CLAPAudioEmbeddingClassifierFreev2(
        pretrained_path=pretrained_path,
        embed_mode="text",
        amodel="HTSAT-tiny",
        unconditional_prob=0.0,
        training_mode=False,
    ).to(device)
    clap.eval()

    T5 = FlanT5HiddenState(text_encoder_name="google/flan-t5-large", freeze_text_encoder=True).to(device)
    T5.eval()

    # Load model from checkpoint using manual module creation and torch.load
    # Instantiate module
    lit_model = LitCLAPT5ToGPT2(
        clap_model=clap,
        T5_model=T5,
        sequence_input_key=["Clap_text_encoder", "T5_text_encoder"],
        sequence_input_embed_dim=[512, 1024],
        target_embed_dim=768,
        max_seq_len=args.max_seq_len,
        mae_token_num=0,
        gpt2_name="gpt2",
    ).to(device)

    # Load checkpoint and filter to inner model weights only (prefixed by 'model.')
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    inner = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = lit_model.model.load_state_dict(inner, strict=False)
    if missing:
        print(f"Warning: Missing keys when loading inner model: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Warning: Unexpected keys when loading inner model: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    lit_model.eval()

    # Data
    dataset = TextFolderDataset(args.input_folder)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_text)

    with torch.no_grad():
        for batch in loader:
            texts = batch["texts"]
            filenames = batch["filenames"]

            # Enforce single item per batch for per-file inference
            if len(texts) != 1:
                for t, f in zip(texts, filenames):
                    single_batch = {"texts": [t], "filenames": [f]}
                    # Compute CLAP and T5
                    clap_emb = clap(single_batch["texts"]).to(device)  # [1, 1, 512]
                    t5_hidden, t5_mask = T5(single_batch["texts"])    # [1, S_t5, 1024], [1, S_t5]
                    t5_hidden = t5_hidden.to(device)
                    t5_mask = t5_mask.to(device)

                    cond_dict = {"Clap_text_encoder": clap_emb, "T5_text_encoder": [t5_hidden, t5_mask]}
                    B, S_t5, _ = t5_hidden.shape

                    # Autoregressive generation up to T5 length
                    gpt2_pred = lit_model.generate(
                        cond_dict=cond_dict,
                        num_tokens=S_t5,
                        device=device,
                    )  # [1, S_t5, 768]

                    out_path = os.path.join(args.output_folder, os.path.splitext(f)[0] + '.npy')
                    out_dict = {
                        'text': single_batch["texts"][0],
                        'gpt2_prediction': gpt2_pred.detach().cpu().numpy()[0],
                        't5_hidden': t5_hidden.detach().cpu().numpy()[0],
                    }
                    np.save(out_path, np.array(out_dict, dtype=object), allow_pickle=True)
                    print(f"Saved: {out_path}")
                continue

            # Compute CLAP and T5
            clap_emb = clap(texts).to(device)  # [1, 1, 512]
            t5_hidden, t5_mask = T5(texts)
            t5_hidden = t5_hidden.to(device)   # [1, S_t5, 1024]
            t5_mask = t5_mask.to(device)       # [1, S_t5]

            # Conditioning dict
            cond_dict = {"Clap_text_encoder": clap_emb, "T5_text_encoder": [t5_hidden, t5_mask]}

            # Autoregressive generation aligned to T5 token length
            gpt2_pred = lit_model.generate(
                cond_dict=cond_dict,
                num_tokens=8,
                device=device,
            )  # [1, S_t5, 768]

            # Save the single item as .npy with a dict
            fname = filenames[0]
            out_path = os.path.join(args.output_folder, os.path.splitext(fname)[0] + '.npy')
            out_dict = {
                'text': texts[0],
                'gpt2_prediction': gpt2_pred.detach().cpu().numpy(),
                't5_hidden': t5_hidden.detach().cpu().numpy(),
                't5_mask': t5_mask.detach().cpu().numpy(),
            }
            np.save(out_path, np.array(out_dict, dtype=object), allow_pickle=True)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
