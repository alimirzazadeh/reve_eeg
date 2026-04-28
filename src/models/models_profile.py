import torch
import torch.nn as nn
from utils.data_loading_h5_mm import (
    DISEASE_MULTI_LEVEL_DICT,
    MEDICATION_MULTI_LEVEL_DICT,
    SEX_DICT,
    RACE_DICT,
    NUM_AGE_GROUP,
)


class ProfileAugment(nn.Module):
    """
    Profile augmentation module that randomly masks a subset of valid
    disease and medication tokens by zeroing their attention mask entries.
    """
    def __init__(self, prob=0.5, min_ratio=0.1):
        super().__init__()
        self.prob = prob
        self.min_ratio = min_ratio

    def forward(self, disease_mask, medication_mask):
        batch_size = disease_mask.shape[0]
        device = disease_mask.device
        disease_length = disease_mask.shape[1]

        augmented_disease_mask = disease_mask.clone()
        augmented_medication_mask = medication_mask.clone()
        combined_mask = torch.cat([disease_mask, medication_mask], dim=1)

        for batch_index in range(batch_size):
            if torch.rand(1, device=device).item() > self.prob:
                continue

            valid_indices = (combined_mask[batch_index] > 0).nonzero(as_tuple=False).flatten()
            valid_count = valid_indices.numel()

            if valid_count <= 1:
                continue

            min_keep = max(1, int(valid_count * self.min_ratio))
            max_mask = valid_count - min_keep
            if max_mask <= 0:
                continue

            num_to_mask = torch.randint(1, max_mask + 1, (1,), device=device).item()
            mask_indices = valid_indices[torch.randperm(valid_count, device=device)[:num_to_mask]]

            disease_indices = mask_indices[mask_indices < disease_length]
            medication_indices = mask_indices[mask_indices >= disease_length] - disease_length

            if disease_indices.numel() > 0:
                augmented_disease_mask[batch_index, disease_indices] = 0
            if medication_indices.numel() > 0:
                augmented_medication_mask[batch_index, medication_indices] = 0

        return augmented_disease_mask, augmented_medication_mask


class ProfileEncoder(nn.Module):
    def __init__(self, 
                 embed_dim=512, 
                 num_layers=4, 
                 nhead=8,
                 disable_multilevel=False,
                 multilevel_embedding_type='sum',
                 disease_level=-1,
                 medication_level=-1,
                 demo_only=False,
                 disable_seg_embed=False,
                 use_profile_augment=False,
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.disable_multilevel = disable_multilevel
        self.disable_seg_embed = disable_seg_embed
        self.multilevel_embedding_type = multilevel_embedding_type
        self.demo_only = demo_only
        self.use_profile_augment = use_profile_augment

        if use_profile_augment:
            self.profile_augment = ProfileAugment()
            print("Using profile augmentation")
        else:
            self.profile_augment = None

        # demo embeddings
        self.age_embedding = nn.Embedding(NUM_AGE_GROUP + 1, embed_dim)
        self.sex_embedding = nn.Embedding(len(SEX_DICT), embed_dim)
        self.race_embedding = nn.Embedding(len(RACE_DICT), embed_dim)

        num_disease_tokens = len(DISEASE_MULTI_LEVEL_DICT['L1'].keys()) + 1
        self.num_disease_levels = len(DISEASE_MULTI_LEVEL_DICT.keys())
        self.disease_level = disease_level
        if not self.disable_multilevel:
            print(f"num_disease_levels: {self.num_disease_levels}",
              f"num_disease_tokens: {num_disease_tokens}")
            if self.multilevel_embedding_type == 'concat':
                # Use embed_dim // num_level for each level
                level_embed_dim = embed_dim // self.num_disease_levels
                self.multi_level_disease_embedding = nn.ModuleList([
                    nn.Embedding(num_disease_tokens, level_embed_dim)
                    for level in range(self.num_disease_levels)
                ])
                # Linear projection to map concatenated embeddings back to embed_dim
                self.disease_projection = nn.Linear(embed_dim, embed_dim)
            else:
                # Original summation approach
                self.multi_level_disease_embedding = nn.ModuleList([
                    nn.Embedding(num_disease_tokens, embed_dim)
                    for level in range(self.num_disease_levels)
                ])
        else:
            print(f"disease_level: {self.disease_level}")
            # When disabled, use a single embedding for the last level (most specific level)
            self.disease_embedding = nn.Embedding(num_disease_tokens, embed_dim)

        num_medication_tokens = len(MEDICATION_MULTI_LEVEL_DICT['L1'].keys()) + 1
        self.num_medication_levels = len(MEDICATION_MULTI_LEVEL_DICT.keys())
        self.medication_level = medication_level
        if not self.disable_multilevel:
            print(f"num_medication_levels: {self.num_medication_levels}",
              f"num_medication_tokens: {num_medication_tokens}")
            if self.multilevel_embedding_type == 'concat':
                # Use embed_dim // num_level for each level
                level_embed_dim = embed_dim // self.num_medication_levels
                self.multi_level_medication_embedding = nn.ModuleList([
                    nn.Embedding(num_medication_tokens, level_embed_dim)
                    for level in range(self.num_medication_levels)
                ])
                # Linear projection to map concatenated embeddings back to embed_dim
                self.medication_projection = nn.Linear(embed_dim, embed_dim)
            else:
                # Original summation approach
                self.multi_level_medication_embedding = nn.ModuleList([
                    nn.Embedding(num_medication_tokens, embed_dim)
                    for level in range(self.num_medication_levels)
                ])
        else:
            print(f"medication_level: {self.medication_level}")
            # When disabled, use a single embedding for the last level (most specific level)
            self.medication_embedding = nn.Embedding(num_medication_tokens, embed_dim)

        if not self.disable_seg_embed:
            # segment (type) embeddings
            self.age_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.sex_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.race_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.disease_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.medication_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(embed_dim)

        # init weights
        torch.nn.init.normal_(self.age_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.sex_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.race_embedding.weight, std=0.02)
        if not self.disable_multilevel:
            for embedding in self.multi_level_disease_embedding:
                torch.nn.init.normal_(embedding.weight, std=0.02)
            for embedding in self.multi_level_medication_embedding:
                torch.nn.init.normal_(embedding.weight, std=0.02)
        else:
            # Initialize the single embedding layers when multilevel is disabled
            torch.nn.init.normal_(self.disease_embedding.weight, std=0.02)
            torch.nn.init.normal_(self.medication_embedding.weight, std=0.02)

        if not self.disable_seg_embed:
            torch.nn.init.normal_(self.age_pos_embed, std=0.02)
            torch.nn.init.normal_(self.sex_pos_embed, std=0.02)
            torch.nn.init.normal_(self.race_pos_embed, std=0.02)
            torch.nn.init.normal_(self.disease_pos_embed, std=0.02)
            torch.nn.init.normal_(self.medication_pos_embed, std=0.02)


    
    def forward(self, demo, disease, disease_mask, medication, medication_mask):
        # demo: [B, 3] (age_token, sex_token, race_token)
        # disease: [B, num_disease_levels, max_num_disease]
        # disease_mask: [B, max_num_disease]
        # medication: [B, num_medication_levels, max_num_medication]
        # medication_mask: [B, max_num_medication]

        age_embed = self.age_embedding(demo[:, 0]).unsqueeze(1) # [B, 1, embed_dim]
        if not self.disable_seg_embed:
            age_embed = age_embed + self.age_pos_embed
        sex_embed = self.sex_embedding(demo[:, 1]).unsqueeze(1)
        if not self.disable_seg_embed:
            sex_embed = sex_embed + self.sex_pos_embed
        race_embed = self.race_embedding(demo[:, 2]).unsqueeze(1)
        if not self.disable_seg_embed:
            race_embed = race_embed + self.race_pos_embed
        
        # Mask all diseases and medications if demo_only is True
        if self.demo_only:
            disease_mask = torch.zeros_like(disease_mask)
            medication_mask = torch.zeros_like(medication_mask)

        if self.use_profile_augment and self.profile_augment is not None:
            disease_mask, medication_mask = self.profile_augment(disease_mask, medication_mask)
        
        if not self.disable_multilevel:
            if self.multilevel_embedding_type == 'concat':
                # Concatenate embeddings from all levels
                level_embeddings = []
                for level in range(self.num_disease_levels):
                    level_embedding = self.multi_level_disease_embedding[level](disease[:, level]) # [B, max_num_disease, level_embed_dim]
                    level_embeddings.append(level_embedding)
                disease_embed = torch.cat(level_embeddings, dim=-1) # [B, max_num_disease, embed_dim]
                # Apply linear projection
                disease_embed = self.disease_projection(disease_embed) # [B, max_num_disease, embed_dim]
            else:
                # Original summation approach
                disease_embed = torch.zeros(disease.shape[0], disease.shape[2], self.embed_dim, device=disease.device)
                for level in range(self.num_disease_levels):
                    level_embedding = self.multi_level_disease_embedding[level](disease[:, level]) # [B, max_num_disease, embed_dim]
                    disease_embed += level_embedding
            if not self.disable_seg_embed:
                disease_embed = disease_embed + self.disease_pos_embed # [B, max_num_disease, embed_dim]
        else:
            # use only the last level embedding (most specific level, e.g., L3 if levels are L1, L2, L3)
            disease_embed = self.disease_embedding(disease[:, self.disease_level]) # [B, max_num_disease, embed_dim]
            if not self.disable_seg_embed:
                disease_embed = disease_embed + self.disease_pos_embed # [B, max_num_disease, embed_dim]

        if not self.disable_multilevel:
            if self.multilevel_embedding_type == 'concat':
                # Concatenate embeddings from all levels
                level_embeddings = []
                for level in range(self.num_medication_levels):
                    level_embedding = self.multi_level_medication_embedding[level](medication[:, level]) # [B, max_num_medication, level_embed_dim]
                    level_embeddings.append(level_embedding)
                medication_embed = torch.cat(level_embeddings, dim=-1) # [B, max_num_medication, embed_dim]
                # Apply linear projection
                medication_embed = self.medication_projection(medication_embed) # [B, max_num_medication, embed_dim]
            else:
                # Original summation approach
                medication_embed = torch.zeros(medication.shape[0], medication.shape[2], self.embed_dim, device=medication.device)
                for level in range(self.num_medication_levels):
                    level_embedding = self.multi_level_medication_embedding[level](medication[:, level]) # [B, max_num_medication, embed_dim]
                    medication_embed += level_embedding
            if not self.disable_seg_embed:
                medication_embed = medication_embed + self.medication_pos_embed # [B, max_num_medication, embed_dim]
        else:
            # use only the last level embedding (most specific level, e.g., L3 if levels are L1, L2, L3)
            medication_embed = self.medication_embedding(medication[:, self.medication_level]) # [B, max_num_medication, embed_dim]
            if not self.disable_seg_embed:
                medication_embed = medication_embed + self.medication_pos_embed # [B, max_num_medication, embed_dim]

        # concatenate all the embeddings
        x = torch.cat([age_embed, sex_embed, race_embed, disease_embed, medication_embed], dim=1) # [B, 1 + 1 + 1 + max_num_disease + max_num_medication, embed_dim]

        # attn valid mask
        demo_mask = torch.ones(x.shape[0], 3, device=x.device, dtype=disease_mask.dtype)
        attn_mask = torch.cat([demo_mask, disease_mask, medication_mask], dim=1)  # [B, 3 + max_num_disease + max_num_medication]
        key_padding_mask = (attn_mask == 0) 

        # Transformer Encoder
        x = self.blocks(x, src_key_padding_mask=key_padding_mask)
        x = self.final_ln(x)

        # pool the valid embeddings
        mask_expanded = attn_mask.unsqueeze(-1) 
        x_masked = x * mask_expanded
        sum_embeddings = x_masked.sum(dim=1)  # [B, D]
        sum_mask = mask_expanded.sum(dim=1)   # [B, 1] count of valid tokens
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        x = sum_embeddings / sum_mask

        # Get pretrained embedding
        one_hot_ehr_vector = self.get_pretrained_embedding(demo, disease, disease_mask, medication, medication_mask)

        return x, one_hot_ehr_vector

    def get_pretrained_embedding(self, demo, disease, disease_mask, medication, medication_mask):
        """
        Returns pretrained embeddings as a concatenated binary vector.
        
        Args:
            demo: [B, 3] (age_token, sex_token, race_token)
            disease: [B, num_disease_levels, max_num_disease]
            disease_mask: [B, max_num_disease]
            medication: [B, num_medication_levels, max_num_medication]
            medication_mask: [B, max_num_medication]
        
        Returns:
            combined_vector: [B, demo_dim + num_diseases + num_medications] - concatenated binary vector
                containing disease_vector, medication_vector, and demo_vector (one-hot for age, sex, race)
                demo_dim = NUM_AGE_GROUP + len(SEX_DICT) + len(RACE_DICT)
        """
        batch_size = demo.shape[0]
        device = demo.device
        
        # Disease vector: binary vector indicating presence of each disease
        num_diseases = len(DISEASE_MULTI_LEVEL_DICT['L1'].keys())
        disease_tokens = disease[:, self.disease_level]  # [B, max_num_disease]
        disease_vector = torch.zeros(batch_size, num_diseases, device=device, dtype=torch.float32)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, disease_tokens.shape[1]).flatten()
        token_indices = disease_tokens.flatten()
        valid_mask = disease_mask.flatten() == 1
        disease_vector[batch_indices[valid_mask], token_indices[valid_mask]] = 1.0
        
        # Medication vector: binary vector indicating presence of each medication
        num_medications = len(MEDICATION_MULTI_LEVEL_DICT['L1'].keys())
        medication_tokens = medication[:, self.medication_level]  # [B, max_num_medication]
        medication_vector = torch.zeros(batch_size, num_medications, device=device, dtype=torch.float32)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, medication_tokens.shape[1]).flatten()
        token_indices = medication_tokens.flatten()
        valid_mask = medication_mask.flatten() == 1
        medication_vector[batch_indices[valid_mask], token_indices[valid_mask]] = 1.0
        
        
        age_onehot = torch.zeros(batch_size, NUM_AGE_GROUP + 1, device=device, dtype=torch.float32)
        batch_indices_age = torch.arange(batch_size, device=device)
        age_onehot[batch_indices_age, demo[:, 0]] = 1.0
        age_onehot = age_onehot[:, :-1]  # Truncate padding class
        
        sex_onehot = torch.zeros(batch_size, len(SEX_DICT), device=device, dtype=torch.float32)
        batch_indices_sex = torch.arange(batch_size, device=device)
        sex_onehot[batch_indices_sex, demo[:, 1]] = 1.0
        sex_onehot = sex_onehot[:, :-1]  # Truncate padding class
        
        race_onehot = torch.zeros(batch_size, len(RACE_DICT), device=device, dtype=torch.float32)
        batch_indices_race = torch.arange(batch_size, device=device)
        race_onehot[batch_indices_race, demo[:, 2]] = 1.0
        race_onehot = race_onehot[:, :-1]  # Truncate padding class
        
        # Concatenate one-hot vectors
        demo_vector = torch.cat([age_onehot, sex_onehot, race_onehot], dim=1)  # [B, NUM_AGE_GROUP + num_sex_classes + num_race_classes]
        
        # Concatenate all three vectors
        combined_vector = torch.cat([demo_vector, disease_vector, medication_vector], dim=1)
        
        return combined_vector
