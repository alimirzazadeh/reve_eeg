import torch
import torch.nn as nn
import numpy as np
from transformers import T5EncoderModel


class TextAugment(nn.Module):
    """
    Text augmentation module that implements random truncation.
    Randomly finds a place to put EOS token (token_idx = 1), makes everything 
    after that pad token (token_idx = 0) and makes the attention mask to be 0.
    """
    def __init__(self, prob=0.5, min_ratio=0.1):
        super().__init__()
        self.prob = prob
        self.min_ratio = min_ratio
    
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Clone to avoid modifying original tensors
        augmented_input_ids = input_ids.clone()
        augmented_attention_mask = attention_mask.clone()
        
        # For each sequence in the batch
        for b in range(batch_size):
            # Randomly decide whether to apply augmentation
            if torch.rand(1, device=device).item() > self.prob:
                continue
            
            # Find the actual length of the sequence (last position with attention_mask = 1)
            valid_length = attention_mask[b].sum().item()

            if valid_length == 1: # empty summary
                continue

            min_length = max(1, int(valid_length * self.min_ratio))
            
            # Randomly select a truncation point between min_length and valid_length-1 (inclusive)
            # This ensures we keep at least min_length tokens
            truncate_at = torch.randint(
                min_length, 
                valid_length, 
                (1,), 
                device=device
            ).item()
            
            # Place EOS token (1) at the truncation point
            augmented_input_ids[b, truncate_at] = 1
            
            # Set everything after truncation point to pad token (0)
            augmented_input_ids[b, truncate_at + 1:] = 0
            
            # Set attention mask to 0 for positions after truncation point
            augmented_attention_mask[b, truncate_at + 1:] = 0
        
        return augmented_input_ids, augmented_attention_mask


class TextEncoder(nn.Module):
    def __init__(self, 
                 model_name="google/t5-v1_1-base", 
                 cache_dir="/data/netmit/RadarFS/Peng/model_weights",
                 context_length=512,
                 embed_dim=512, 
                 num_layers=4, 
                 nhead=8,
                 use_text_augment=False,
                 aggregation='eos'):
        super().__init__()
        
        print(f"Loading T5 encoder from {cache_dir}")
        self.t5_encoder = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir)
        # Freeze T5 encoder parameters (if you want to train T5, remove these lines)
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        self.t5_dim = self.t5_encoder.config.d_model
        
        # Text augmentation module
        self.use_text_augment = use_text_augment
        if use_text_augment:
            self.text_augment = TextAugment()
            print("Using text augmentation")
        else:
            self.text_augment = None
        
        # Projection to your desired embedding dimension
        self.input_proj = nn.Linear(self.t5_dim, embed_dim)
        
        # Standard learned Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.aggregator = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.aggregation = aggregation
        self.final_ln = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask):
        # 0. Apply text augmentation if enabled (only during training)
        if self.use_text_augment and self.text_augment is not None:
            input_ids, attention_mask = self.text_augment(input_ids, attention_mask)

        assert (input_ids == 1).sum() == input_ids.shape[0], "There should be exactly one EOS token in each sequence"
        
        # 1. Get T5 Encoder hidden states
        with torch.no_grad():
            self.t5_encoder.eval()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                t5_outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_features = t5_outputs.last_hidden_state # [B, Seq_Len, 768]
        
        # Extract global feature from T5 encoder (with no gradient)
        # Mean pooling over non-padding tokens from T5 encoder output
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, Seq_Len, 1]
        masked_t5_features = sequence_features * mask_expanded  # [B, Seq_Len, t5_dim]
        sum_t5_features = masked_t5_features.sum(dim=1)  # [B, t5_dim]
        token_counts = attention_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
        token_counts = torch.clamp(token_counts, min=1e-9)
        global_feature_pretrained = (sum_t5_features / token_counts).detach()  # [B, t5_dim], no gradient
        
        # 2. Project and add Positional Embeddings
        x = self.input_proj(sequence_features)
        # Match pos_embed to the current sequence length (for flexibility)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # 3. Transformer Aggregator 
        # src_key_padding_mask expects True for padding positions
        key_padding_mask = (attention_mask == 0) 
        
        # Pass through the aggregator transformer
        # [B, Seq_Len, embed_dim]
        refined_features = self.aggregator(x, src_key_padding_mask=key_padding_mask)
        
        # 4. Apply final layer norm to all tokens (consistent with ProfileEncoder)
        refined_features = self.final_ln(refined_features)
        
        # 5. Aggregate features based on aggregation method
        if self.aggregation == 'mean':
            # Mean pooling over non-padding tokens
            # attention_mask: 1 for valid tokens, 0 for padding
            # Expand attention_mask to match feature dimensions for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, Seq_Len, 1]
            # Sum features over non-padding tokens
            masked_features = refined_features * mask_expanded  # [B, Seq_Len, embed_dim]
            sum_features = masked_features.sum(dim=1)  # [B, embed_dim]
            # Count non-padding tokens per sequence
            token_counts = attention_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            # Avoid division by zero (though this shouldn't happen with valid sequences)
            token_counts = torch.clamp(token_counts, min=1e-9)
            # Compute mean
            aggregated_features = sum_features / token_counts  # [B, embed_dim]
        else:
            # Extract the EOS token representation
            # In T5, the EOS token (ID 1) is usually the last non-padded token.
            # We find the index of the last '1' in each sequence.
            eos_indices = (input_ids == 1).nonzero(as_tuple=True)[1]
            
            # Pull the features at the EOS positions
            # batch_idx = [0, 1, 2...], eos_idx = [idx_of_1, idx_of_1...]
            batch_indices = torch.arange(x.shape[0], device=x.device)
            aggregated_features = refined_features[batch_indices, eos_indices]

        return aggregated_features, global_feature_pretrained