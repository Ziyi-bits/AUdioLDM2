import torch
import torch.nn as nn


import sys
sys.path.append(r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning")
from audioldm_train.conditional_models import CLAPAudioEmbeddingClassifierFreev2

class CLAPToGPT2Adapter(nn.Module):
    """
    Maps one or more input sequences (e.g., CLAP embeddings) to GPT-2 dimension (768),
    adds SOS/EOS tokens, builds attention masks, and truncates to GPT-2 context length.
    """

    def __init__(self,
                 sequence_input_key,            # list of keys, e.g., ["audio", "text"]
                 sequence_input_embed_dim,      # list of dims, e.g., [512, 512]
                 target_embed_dim=768,
                 max_seq_len=1024,
                 mae_token_num=0,
                 target_tokens_mask_ratio=0.0):
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
                mask = torch.ones((seq.size(0), seq.size(1)), device=seq.device)

            seq = self.input_sequence_embed_linear[_id](seq)  # project
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

if __name__ == "__main__":
    # Path to your pretrained CLAP checkpoint
    pretrained_path = r"C:\Users\ZiXu\Documents\Python_Scripts\Git\AudioDLM2\AudioLDM-training-finetuning\data\checkpoints\clap_htsat_tiny.pt"
    pretrained_path = pretrained_path.replace("\\", "/")


    # 1) Initialize model in TEXT mode
    model = CLAPAudioEmbeddingClassifierFreev2(
        pretrained_path=pretrained_path,
        embed_mode="text",       # start with text
        amodel="HTSAT-tiny",
        unconditional_prob=0.0,  # disable CFG masking for test
        training_mode=False
    )

    print(">>> Testing TEXT mode...")
    text_data = ["a dog barking loudly", "a piano playing a melody"]  # batch of 2

    clap_emb = model(text_data)  # shape [B, 1, 512]
    # mask = torch.ones((clap_emb.size(0), clap_emb.size(1)), device=clap_emb.device)
    print("Raw CLAP text embedding shape:", clap_emb.shape)  # [B, 1, 512]
    # print("Raw CLAP text embedding mask shape:", mask.shape)  # [B, 1]

    # 2) Initialize adapter
    adapter = CLAPToGPT2Adapter(
        sequence_input_key=["Clap_text_encoder"],
        sequence_input_embed_dim=[512],  # CLAP outputs 512-d
        target_embed_dim=768,
        max_seq_len=1024
    )

    # 3) Prepare cond_dict and map to GPT-2 dimension
    cond_dict = {"Clap_text_encoder": clap_emb}
    gpt2_inputs, attn_mask, end_idx = adapter.get_input_sequence_and_mask(cond_dict)

    print("end_idx:", end_idx)

    print("GPT-2 input shape:", gpt2_inputs.shape)  # [B, T+2, 768]
    print("Attention mask shape:", attn_mask.shape)  # [B, T+2]


