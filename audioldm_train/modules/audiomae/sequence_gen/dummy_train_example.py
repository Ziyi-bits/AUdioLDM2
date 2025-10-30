import os
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Model
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import sys
import numpy as np
sys.path.append(r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning")
from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2
# ---------------------------
# Reproducibility (optional)
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# ---------------------------
# Dummy dataset
# ---------------------------
class DummyAudioDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "text": "Dummy text data",
            "crossattn_audiomae_pooled": (torch.randn(8, 768), torch.ones(8))  # target (S_tgt, H), mask (S_tgt,)
            # "crossattn_audiomae_mask": (torch.ones(8))  # target (S_tgt, H), mask (S_tgt,)
        }

# ---------------------------
# Test dataset
# ---------------------------
class TestAudioDataset(Dataset):
    # initialize with your test data

    @staticmethod
    def get_list_of_files(directory, extension):
        """Return sorted list of files with given extension."""
        return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

    def __init__(self,
                 num_samples=10,
                 data_path="path/to/your/test/data",
                 text_file_extension=".txt",
                 embed_file_extension=".npy",
                 text_file_path="path/to/your/test/texts",
                 embed_file_path="path/to/your/test/embeddings",
                 attention_file_path="path/to/your/test/attention_masks"):
        self.data_path = data_path
        self.text_file_extension = text_file_extension
        self.embed_file_extension = embed_file_extension
        self.text_file_path = text_file_path
        self.embed_file_path = embed_file_path
        self.attention_file_path = attention_file_path
        self.text_file_path = os.path.join(data_path, 'texts')
        self.embed_file_path = os.path.join(data_path, 'mae_embeds')
        self.attention_file_path = os.path.join(data_path, 'mae_attn_mask')

        # check number of .txt files in data_path
        num_txt_files = len([f for f in os.listdir(self.text_file_path) if f.endswith('.txt')])
        # check number of .npy files in embed_file_path
        num_npy_files = len([f for f in os.listdir(self.embed_file_path) if f.endswith('.npy')])
        # check number of .npy files in attention_file_path
        num_attn_files = len([f for f in os.listdir(self.attention_file_path) if f.endswith('.npy')])
        assert num_txt_files == num_npy_files == num_attn_files, "Mismatch in number of text, embedding, or attention mask files"
        self.num_samples = num_txt_files
        print("Dataset initialize finished")

        self.text_files = self.get_list_of_files(self.text_file_path, self.text_file_extension)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load text
        text_file = self.text_files[idx]
        with open(os.path.join(self.text_file_path, text_file), 'r') as f:
            text_data = f.read().strip()
        file_id = os.path.splitext(text_file)[0]
        # Load embedding
        embed_file = file_id + self.embed_file_extension
        #read numpy array
        numpy_struct = np.load(os.path.join(self.embed_file_path, embed_file))
        embed_data = torch.from_numpy(numpy_struct).float()  # [S_tgt, H]
        # Load attention mask
        attention_file = file_id + self.embed_file_extension
        attention_struct = np.load(os.path.join(self.attention_file_path, attention_file))
        attention_mask = torch.from_numpy(attention_struct).float()  # [S_tgt]

        return {
            "text": text_data,
            "crossattn_audiomae_pooled": (embed_data, attention_mask)  # target (S_tgt, H), mask (S_tgt,)
        }

# ---------------------------
# GPT2 wrapper
# ---------------------------
class GPT2SequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2Model.from_pretrained("gpt2")  # hidden_size = 768
    def forward(self, inputs_embeds, attention_mask):
        # inputs_embeds: (B, S, H), attention_mask: (B, S)
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)["last_hidden_state"]


# ---------------------------
# CLAP-to-GPT2 adapter logic and GPT-2 backbone into one trainable model
# ---------------------------
class CLAPToGPT2(nn.Module):
    """
    Combines CLAP-to-GPT2 adapter logic and GPT-2 backbone into one trainable model.
    Handles:
      - Projection of conditioning sequences to GPT-2 hidden size
      - Adding SOS/EOS tokens
      - Building attention masks
      - Truncating to GPT-2 context length
      - Forward pass through GPT-2
      - Returning outputs aligned for next-token prediction
    """

    def __init__(self,
                 sequence_input_key,            # list of keys, e.g., ["audio", "text"]
                 sequence_input_embed_dim,      # list of dims, e.g., [512, 512]
                 target_embed_dim=768,
                 max_seq_len=1024,
                 mae_token_num=0,
                 target_tokens_mask_ratio=0.0,
                 gpt2_name="gpt2"):
        super().__init__()
        assert len(sequence_input_key) == len(sequence_input_embed_dim), \
            "Keys and dims must match in length"

        self.sequence_input_key = sequence_input_key
        self.sequence_input_embed_dim = sequence_input_embed_dim
        self.target_tokens_mask_ratio = target_tokens_mask_ratio
        self.target_embed_dim = target_embed_dim
        self.max_seq_len = max_seq_len
        self.mae_token_num = mae_token_num

        # Learned SOS/EOS embeddings (up to 32 sequence types)
        self.start_of_sequence_tokens = nn.Embedding(32, target_embed_dim)
        self.end_of_sequence_tokens = nn.Embedding(32, target_embed_dim)

        # Linear projections for each input type
        self.input_sequence_embed_linear = nn.ModuleList([
            nn.Linear(dim, target_embed_dim) for dim in sequence_input_embed_dim
        ])

        # GPT-2 backbone
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)

        trainable_params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        print(f"Trainable parameters in GPT-2: {trainable_params:,}")

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        """
        Adds SOS/EOS tokens and updates attention mask.
        sequence: [B, T, D], attn_mask: [B, T]
        """
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
            print(f"Truncating sequence from {sequence.size(1)} to {max_len}")
            return sequence[:, :max_len], mask[:, :max_len]
        return sequence, mask

    def get_input_sequence_and_mask(self, cond_dict):
        """
        cond_dict: { key: Tensor [B,T,D] or [Tensor, mask] }
        Returns: input_embeds [B,S,768], attn_mask [B,S], cond_end_idx
        """
        input_embeds, input_mask = None, None

        for _id, key in enumerate(self.sequence_input_key):
            assert key in cond_dict, f"Missing key {key}"
            cond_embed = cond_dict[key]

            if isinstance(cond_embed, list):
                seq, mask = cond_embed
            else:
                seq = cond_embed
                if seq.dim() == 2:  # [B, D] -> [B, 1, D]
                    seq = seq.unsqueeze(1)
                mask = torch.ones((seq.size(0), seq.size(1)), device=seq.device)

            seq = self.input_sequence_embed_linear[_id](seq) #project
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
        """
        cond_dict: dict of conditioning sequences
        target_embeds: [B, S_tgt, H]
        target_mask: [B, S_tgt]
        Returns: GPT-2 outputs aligned with target tokens [B, S_tgt, H]
        """
        input_embeds, input_mask, cond_end_idx = self.get_input_sequence_and_mask(cond_dict)

        # Concatenate context + target
        final_input_embeds = torch.cat([input_embeds, target_embeds], dim=1)
        final_input_mask = torch.cat([input_mask, target_mask], dim=1)

        # GPT-2 forward
        output_embeds = self.gpt2(
            inputs_embeds=final_input_embeds,
            attention_mask=final_input_mask.long()
        )["last_hidden_state"]

        # Slice outputs for next-token prediction
        output = output_embeds[:, cond_end_idx - 1 : -1, :]  # [B, S_tgt, H]
        return output


# ---------------------------
# Loss function
# ---------------------------
def compute_loss(output, target, target_mask =None):
    """
    output: [B, S_tgt, H]
    target: [B, S_tgt, H]
    """
    if target_mask is None:
        loss_val = nn.L1Loss()(output, target)
    else:
        m = target_mask.float().unsqueeze(-1)           # [B, S, 1]
        l1 = torch.abs(output - target) * m             # [B, S, H]
        denom = (m.sum() * output.size(-1)).clamp_min(1.0)
        loss_val = l1.sum() / denom
    return loss_val


# ---------------------------
# Warm-up logic
# ---------------------------
def warmup_learning_rate(optimizer, step, warmup_steps, base_lr):
    if step < warmup_steps:  # fixed '<'
        lr = base_lr * float(step) / float(warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# ---------------------------
# Evaluation loop
# ---------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        # 1) Build cond_dict from CLAP text encoder (no grad)
        input_text = batch["text"]  # list[str] of length B
        with torch.no_grad():
            clap_emb = model_clap(input_text).to(device)  # [B, 1, 512]

        cond_dict = {"Clap_text_encoder": clap_emb}

        # 2) Targets
        target_embeds, target_mask = batch["crossattn_audiomae_pooled"]
        target_embeds = target_embeds.to(device)          # [B, S_tgt, 768]
        target_mask = target_mask.to(device).float()    # [B, S_tgt]

        # 3) Forward through integrated model (adapter + GPT-2)
        output = model(cond_dict, target_embeds, target_mask)  # [B, S_tgt, 768]

        # 4) Loss (unmasked, matches original)
        loss = compute_loss(output, target_embeds)

        total_loss += loss.item()
        total_batches += 1

    avg_val_loss = total_loss / max(1, total_batches)
    return avg_val_loss

# ---------------------------
# Training loop with validation, checkpoint, early stopping
# ---------------------------
def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=5,
    warmup_steps=1000,
    base_lr=1e-4,
    patience=5,
    ckpt_path="best_model.pt",
    log_every=100,
    scheduler_type= "CosineAnnealingLR",
    step_size=10,  # for StepLR
    gamma=0.8,  # for StepLR
    T_max=None  # for CosineAnnealingLR (defaults to num_epochs if None)

):
    model.to(device)
    step = 0
    best_val_loss = math.inf
    patience_count = 0


    # ---- Scheduler Initialization ----
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Using StepLR: step_size={step_size}, gamma={gamma}")
    elif scheduler_type == "CosineAnnealingLR":
        if T_max is None:
            T_max = num_epochs  # default to total epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
        print(f"Using CosineAnnealingLR: T_max={T_max}, eta_min=1e-6")
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            # 1) Build cond_dict from CLAP text encoder (no grad)
            input_text = batch["text"]  # list[str] of length B
            with torch.no_grad():
                clap_emb = model_clap(input_text).to(device)  # [B, 1, 512]
            cond_dict = {"Clap_text_encoder": clap_emb}

            # 2) Targets
            target_embeds, target_mask = batch["crossattn_audiomae_pooled"]
            target_embeds = target_embeds.to(device)        # [B, S_tgt, 768]
            target_mask = target_mask.to(device).float()  # [B, S_tgt]

            # 3) Forward/backward
            optimizer.zero_grad(set_to_none=True)
            warmup_learning_rate(optimizer, step, warmup_steps, base_lr)

            output = model(cond_dict, target_embeds, target_mask)  # [B, S_tgt, 768]

            # Unmasked L1 (matches original)
            loss = compute_loss(output, target_embeds)
            # If you want masked loss later:
            # loss = masked_l1_loss(output, target_embeds, target_mask)

            loss.backward()
            optimizer.step()

            # 4) Logging
            epoch_loss += loss.item()
            batches += 1
            if step % log_every == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"[Train] Epoch {epoch} Step {step} | LR: {cur_lr:.6e} | Loss: {loss.item():.4f}")

            step += 1

        avg_train_loss = epoch_loss / max(1, batches)

        # ---- Validation at end of epoch ----
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} completed | Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Step the scheduler once per epoch
        scheduler.step()

        # ---- Checkpointing + Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0

            # Save a robust checkpoint including optimizer & scheduler states
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "train_loss": avg_train_loss
            }, ckpt_path)
            print(f"  → New best model saved to '{ckpt_path}' (val_loss={val_loss:.4f}, train_loss={avg_train_loss:.4f}, epoch={epoch})")
        else:
            patience_count += 1
            print(f"  → No improvement. Early stop counter: {patience_count}/{patience}")
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training done. Best Val Loss: {best_val_loss:.4f}. Training Loss: {avg_train_loss:.4f}. Model saved to '{ckpt_path}'")
    return best_val_loss, avg_train_loss


# ---------------------------
# Setup
# ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = r"C:\Users\ZiXu\Documents\Python_Scripts\mae_output"
data_path = data_path.replace("\\", "/")
full_dataset = TestAudioDataset(data_path=data_path)

# Split into train / val (e.g., 80/20)
val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
batch_size = 2 # your desired batch size

# Keep batch_size=1 to match sequence concat logic
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the adapter (example)
pretrained_path = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\data\checkpoints\clap_htsat_tiny.pt"
pretrained_path = pretrained_path.replace("\\", "/")
# Initialize CLAP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_clap = CLAPAudioEmbeddingClassifierFreev2(
    pretrained_path=pretrained_path,
    embed_mode="text",  # start with text
    amodel="HTSAT-tiny",
    unconditional_prob=0.0,  # disable CFG masking for test
    training_mode=False
).to(device)
model_clap.eval()  # Ensures inference mode

# Initialize CLAP-to-GPT2 adapter model
model = CLAPToGPT2(
    sequence_input_key=["Clap_text_encoder"],
    sequence_input_embed_dim=[512],   # CLAP text embedding dim
    target_embed_dim=768,
    max_seq_len=1024,
    mae_token_num=0,                  # leave space for targets if needed in your full code
    gpt2_name="gpt2"
).to(device)



def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage:
print(f"Trainable parameters: {count_trainable_params(model):,}")


optimizer = AdamW(model.parameters(), lr=1e-4)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.8)  # stepped per epoch in train()
# scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# ---------------------------
# Train with validation, checkpointing, early stopping
# ---------------------------
P = 5  # patience
best_val, train_loss = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=30,
    warmup_steps=1000,
    base_lr=1e-4,
    patience=P,
    ckpt_path="best_model.pt",
    log_every=4,
    scheduler_type="CosineAnnealingLR",
    T_max=None
)

# ---------------------------
# (Optional) How to load the best checkpoint later
# ---------------------------
# ckpt = torch.load("best_model.pt", map_location=device)
# model.load_state_dict(ckpt["model_state_dict"])
# optimizer.load_state_dict(ckpt["optimizer_state_dict"])
# scheduler.load_state_dict(ckpt["scheduler_state_dict"])
# start_epoch = ckpt["epoch"] + 1
# print("Resumed from best checkpoint with val_loss:", ckpt["val_loss"])