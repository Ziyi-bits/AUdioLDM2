# ============================
# Lightning implementation
# ============================
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Model
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import sys
import numpy as np
sys.path.append(r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning")
from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2, FlanT5HiddenState
from transformers import get_cosine_schedule_with_warmup

# ckpt = torch.load("./best_model.ckpt")
# print(ckpt.keys())
# # print out the optimzer state dict keys AND THE lr scheduler state dict keys
# print(ckpt['optimizer_states'][0].keys())
# print(ckpt['lr_schedulers'][0]['_last_lr'])
# ============================
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
        # if embed_data has more than 2 dimentions, squeeze it and remove the first two dimension
        if embed_data.dim() > 2:
            embed_data = embed_data.squeeze(0)
        # Load attention mask
        attention_file = file_id + self.embed_file_extension
        attention_struct = np.load(os.path.join(self.attention_file_path, attention_file))
        # if attention_struct has more than 1 dimentions, squeeze it and remove the first dimension
        if attention_struct.ndim > 1:
            attention_struct = attention_struct.squeeze(0)
        attention_mask = torch.from_numpy(attention_struct).long()  # [S_tgt]

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
class CLAPT5ToGPT2(nn.Module):
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

class LitCLAPT5ToGPT2(pl.LightningModule):
    def __init__(
        self,
        clap_model,                          # frozen CLAP (nn.Module)
        T5_model,                           # frozen T5 (nn.Module)
        sequence_input_key,
        sequence_input_embed_dim,
        target_embed_dim=768,
        max_seq_len=1024,
        mae_token_num=0,
        gpt2_name="gpt2",
        # optim/schedule
        base_lr=1e-4,
        warmup_steps=1000,
        scheduler_type="CosineAnnealingLR",  # accept: "CosineAnnealingLR" | "HF_CosineWithWarmup" | "StepLR"
        step_size=10,                        # for StepLR
        gamma=0.8,                           # for StepLR
        T_max=None,                          # for CosineAnnealingLR
        eta_min=1e-6,
        num_training_steps=None,
        # logging
        log_every_n_steps=50,
    ):
        super().__init__()
        # Save hparams but ignore the large external module in checkpoints
        self.save_hyperparameters(ignore=["clap_model", "T5_model"])

        # ---- CLAP (frozen, eval) ----
        self.clap = clap_model.eval()
        for p in self.clap.parameters():
            p.requires_grad = False
        self.T5 = T5_model.eval()
        for p in self.T5.parameters():
            p.requires_grad = False

        # ---- Core model (trainable) ----
        self.model = CLAPT5ToGPT2(
            sequence_input_key=sequence_input_key,
            sequence_input_embed_dim=sequence_input_embed_dim,
            target_embed_dim=target_embed_dim,
            max_seq_len=max_seq_len,
            mae_token_num=mae_token_num,
            gpt2_name=gpt2_name,
        )

        self.loss_fn = nn.L1Loss()
        self._base_lr = float(base_lr)
        self._warmup_steps = int(warmup_steps)
        self._scheduler_type = scheduler_type
        self._step_size = step_size
        self._gamma = gamma
        self._T_max = T_max
        self._eta_min = eta_min
        self._log_every_n_steps = log_every_n_steps
        self._use_hf_warmup = (scheduler_type == "HF_CosineWithWarmup")
        self._num_training_steps = num_training_steps
    # --------- Utility ----------
    def _linear_warmup(self, optimizer):
        """Step-based linear warmup using self.global_step."""
        if self.global_step < self._warmup_steps:
            new_lr = self._base_lr * (float(self.global_step) / float(self._warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

    # --------- Forward ----------
    def forward(self, cond_dict, target_embeds, target_mask):
        return self.model(cond_dict, target_embeds, target_mask)

    # --------- Training ----------
    def training_step(self, batch, batch_idx):
        # 1) Build cond_dict from CLAP (no grad, eval semantics)
        input_text = batch["text"]  # list[str]
        with torch.no_grad():
            clap_emb = self.clap(input_text).to(self.device)  # [B, 1, 512] (as per your CLAP wrapper)
            t5_hidden, t5_mask = self.T5(input_text)  # FlanT5HiddenState returns [hidden_state, mask]

            t5_hidden = t5_hidden.to(self.device)
            t5_mask = t5_mask.to(self.device)

        cond_dict = {"Clap_text_encoder": clap_emb, "T5_text_encoder": [t5_hidden, t5_mask]}
        # 2) Targets
        target_embeds, target_mask = batch["crossattn_audiomae_pooled"]
        target_embeds = target_embeds.to(self.device)      # [B, S_tgt, 768]
        target_mask = target_mask.to(self.device).float()  # [B, S_tgt]

        # 3) Forward & loss (unmasked L1 to match your script)
        output = self(cond_dict, target_embeds, target_mask)  # [B, S_tgt, 768]
        loss = self.loss_fn(output, target_embeds)

        # 4) Apply step-based warmup (overrides scheduler during warmup)
        optim = self.optimizers()
        if not self._use_hf_warmup:
            self._linear_warmup(optim)

        # Log LR every n steps and also aggregate per epoch
        if self.global_step % self._log_every_n_steps == 0:
            cur_lr = optim.param_groups[0]["lr"]
            self.log("train_lr", cur_lr, on_step=True, on_epoch=True, prog_bar=True)
        # Log loss only per epoch
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # --------- Validation ----------
    def validation_step(self, batch, batch_idx):
        input_text = batch["text"]
        with torch.no_grad():
            clap_emb = self.clap(input_text).to(self.device)
            t5_hidden, t5_mask = self.T5(input_text)  # FlanT5HiddenState returns [hidden_state, mask]

            t5_hidden = t5_hidden.to(self.device)
            t5_mask = t5_mask.to(self.device)

        cond_dict = {"Clap_text_encoder": clap_emb, "T5_text_encoder": [t5_hidden, t5_mask]}

        target_embeds, target_mask = batch["crossattn_audiomae_pooled"]
        target_embeds = target_embeds.to(self.device)
        target_mask = target_mask.to(self.device).float()

        output = self(cond_dict, target_embeds, target_mask)
        val_loss = self.loss_fn(output, target_embeds)

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return val_loss

    # --------- Optim / Schedulers ----------
    def configure_optimizers(self):
        # Only optimize the trainable core model (not CLAP)
        optimizer = AdamW(self.model.parameters(), lr=self._base_lr)

        # Epoch-level scheduler (Lightning will call .step() each epoch)
        if self._scheduler_type == "StepLR":
            scheduler = StepLR(optimizer, step_size=self._step_size, gamma=self._gamma)
            step_type = "epoch"
        elif self._scheduler_type == "CosineAnnealingLR":
            T_max = self._T_max if self._T_max is not None else self.trainer.max_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=self._eta_min)
            step_type = "epoch"
        elif self._scheduler_type == "HF_CosineWithWarmup":
            if self._num_training_steps is None:
                # Prefer Lightning’s estimate (PL 2.0+)
                if hasattr(self.trainer, "estimated_stepping_batches") and self.trainer.estimated_stepping_batches:
                    total_steps = int(self.trainer.estimated_stepping_batches)
                else:
                    raise ValueError("num_training_steps was not provided and could not be inferred from Trainer.")
            else:
                total_steps = int(self._num_training_steps)

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self._warmup_steps),
                num_training_steps=total_steps,
            )
            step_type = "step"
        else:
            raise ValueError(f"Unsupported scheduler_type: {self._scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": step_type,   # step the scheduler every epoch. If using HF_CosineWithWarmup, use "step"!!!!!!
                "frequency": 1,
                "monitor": "val_loss"  # used by EarlyStopping/ModelCheckpoint if configured
            },
        }

    # (Optional) keep checkpoints light by not saving CLAP and T5 weights
    def on_save_checkpoint(self, checkpoint):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            drop_prefixes = ("clap.", "T5.")
            keys_to_drop = [k for k in list(checkpoint["state_dict"].keys()) if k.startswith(drop_prefixes)]
            for k in keys_to_drop:
                checkpoint["state_dict"].pop(k, None)

# ============================
# Build datasets & dataloaders
# ============================
data_path = r"C:\Users\ZiXu\Documents\Python_Scripts\mae_output_new".replace("\\", "/")
full_dataset = TestAudioDataset(data_path=data_path)

val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

# ============================
# CLAP (frozen, eval)
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_path = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\data\checkpoints\clap_htsat_tiny.pt".replace("\\", "/")

model_clap = CLAPAudioEmbeddingClassifierFreev2(
    pretrained_path=pretrained_path,
    embed_mode="text",
    amodel="HTSAT-tiny",
    unconditional_prob=0.0,
    training_mode=False,
).to(device)
model_clap.eval()  # deterministic inference

model_T5 = FlanT5HiddenState(text_encoder_name="google/flan-t5-large", freeze_text_encoder=True).to(device)
model_T5.eval()  # deterministic inference


# ============================
# Lightning module
# ============================
max_epochs = 30
total_steps = max_epochs * len(train_loader)
warmup_steps = int(0.1 * total_steps)

lit_model = LitCLAPT5ToGPT2(
    clap_model=model_clap,
    T5_model=model_T5,
    sequence_input_key=["Clap_text_encoder", "T5_text_encoder"],
    sequence_input_embed_dim=[512, 1024],    # CLAP text embedding dim and T5 text embedding dim
    target_embed_dim=768,
    max_seq_len=1024,
    mae_token_num=0,
    gpt2_name="gpt2",
    # training config
    base_lr=1e-4,
    warmup_steps=warmup_steps,
    scheduler_type="HF_CosineWithWarmup",  # or "StepLR"
    step_size=10,
    gamma=0.8,
    T_max=None,     # default to max_epochs
    eta_min=1e-6,
    log_every_n_steps=4,
    num_training_steps=total_steps
)

# ============================
# Callbacks: checkpoint + early stop
# ============================
checkpoint_cb = ModelCheckpoint(
    dirpath=".",
    filename="best_model",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,  # P
    verbose=True,
)

# ============================
# Trainer
# ============================
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices="auto",
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    log_every_n_steps=4,
    callbacks=[checkpoint_cb, early_stop_cb],
)

# ============================
# Fit
# ============================
trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)