"""Multimodal CLIP-style encoder using REVE as the EEG vision backbone."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from models import models_profile, models_text
from models.encoder import REVE


class MultiModalEncoder(nn.Module):
    """
    Vision: REVE encoder. The REVE forward returns tokens of shape
    [B, C * H, embed_dim] (C = number of EEG channels, H = number of time
    patches per channel). We pool over the time axis and concatenate the
    channels into a single [B, C * embed_dim] image embedding before the
    image-text/profile projections.
    """

    def __init__(
        self,
        # vision (REVE)
        reve_embed_dim: int = 512,
        reve_depth: int = 22,
        reve_heads: int = 8,
        reve_head_dim: int = 64,
        reve_mlp_dim_ratio: float = 2.66,
        reve_use_geglu: bool = True,
        reve_freqs: int = 4,
        reve_patch_size: int = 200,
        reve_patch_overlap: int = 20,
        reve_noise_ratio: float = 0.0025,
        reve_num_channels: int = 19,
        reve_checkpoint_path: str | None = None,
        time_pool: str = "mean",
        # text
        text_encoder_num_layers: int = 4,
        text_encoder_nhead: int = 8,
        text_model_name: str = "google/t5-v1_1-base",
        text_model_cache_dir: str = "/data/netmit/RadarFS/Peng/model_weights",
        text_model_context_length: int = 256,
        use_text_augment: bool = False,
        text_aggregation: str = "eos",
        # profile
        profile_encoder_num_layers: int = 4,
        profile_encoder_nhead: int = 8,
        disable_profile_multilevel: bool = False,
        profile_multilevel_embedding_type: str = "sum",
        profile_demo_only: bool = False,
        use_profile_augment: bool = False,
        disease_level: int = -1,
        medication_level: int = -1,
        disable_seg_embed: bool = False,
        # vl
        text_width: int = 512,
        profile_width: int = 512,
        vl_projection: str = "linear",
        mlp_dim: int = 2048,
        emb_dim: int = 512,
        disable_clip: bool = False,
        disable_profile: bool = False,
        disable_text: bool = False,
    ):
        super().__init__()

        self.disable_clip = disable_clip
        self.disable_profile = disable_profile
        self.disable_text = disable_text

        # ---- vision: REVE encoder ----
        backbone_args = SimpleNamespace(
            embed_dim=reve_embed_dim,
            depth=reve_depth,
            heads=reve_heads,
            head_dim=reve_head_dim,
            mlp_dim_ratio=reve_mlp_dim_ratio,
            use_geglu=reve_use_geglu,
        )
        self.visual = REVE(
            args_backbone=backbone_args,
            freqs=reve_freqs,
            patch_size=reve_patch_size,
            overlap_size=reve_patch_overlap,
            noise_ratio=reve_noise_ratio,
        )
        self.reve_num_channels = reve_num_channels
        self.reve_embed_dim = reve_embed_dim
        self.time_pool = time_pool

        if reve_checkpoint_path is not None:
            self._load_reve_weights(reve_checkpoint_path)

        # vision embedding dimension after time-pooling + channel concat
        vision_width = reve_num_channels * reve_embed_dim

        # ---- text ----
        self.text_encoder = models_text.TextEncoder(
            model_name=text_model_name,
            cache_dir=text_model_cache_dir,
            context_length=text_model_context_length,
            embed_dim=text_width,
            num_layers=text_encoder_num_layers,
            nhead=text_encoder_nhead,
            use_text_augment=use_text_augment,
            aggregation=text_aggregation,
        )

        # ---- profile ----
        self.profile_encoder = models_profile.ProfileEncoder(
            embed_dim=profile_width,
            num_layers=profile_encoder_num_layers,
            nhead=profile_encoder_nhead,
            disable_multilevel=disable_profile_multilevel,
            multilevel_embedding_type=profile_multilevel_embedding_type,
            demo_only=profile_demo_only,
            use_profile_augment=use_profile_augment,
            disease_level=disease_level,
            medication_level=medication_level,
            disable_seg_embed=disable_seg_embed,
        )

        # ---- vision-language projections ----
        self.vl_projection = vl_projection
        if self.vl_projection == "linear":
            self.image_text_projection = nn.Parameter(torch.empty(vision_width, emb_dim))
            self.image_profile_projection = nn.Parameter(torch.empty(vision_width, emb_dim))
            self.text_projection = nn.Parameter(torch.empty(text_width, emb_dim))
            self.profile_projection = nn.Parameter(torch.empty(profile_width, emb_dim))
        elif self.vl_projection == "mlp":
            self.image_text_projection = self._build_mlp(vision_width, mlp_dim, emb_dim)
            self.image_profile_projection = self._build_mlp(vision_width, mlp_dim, emb_dim)
            self.text_projection = self._build_mlp(text_width, mlp_dim, emb_dim)
            self.profile_projection = self._build_mlp(profile_width, mlp_dim, emb_dim)
        else:
            raise ValueError(f"Invalid vl_projection: {self.vl_projection}")

        self.vision_width = vision_width
        self.text_width = text_width
        self.profile_width = profile_width
        self.emb_dim = emb_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_profile = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_weights()

    def _load_reve_weights(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Strip optional "encoder." / "module." prefixes that may show up in saved checkpoints
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            new_k = k
            for prefix in ("module.", "encoder."):
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
            cleaned[new_k] = v

        missing, unexpected = self.visual.load_state_dict(cleaned, strict=False)
        print(
            f"Loaded REVE weights from {ckpt_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing:
            print(f"  first missing keys: {missing[:5]}")
        if unexpected:
            print(f"  first unexpected keys: {unexpected[:5]}")

    def initialize_weights(self):
        if self.vl_projection == "linear":
            nn.init.normal_(self.image_text_projection, std=self.vision_width ** -0.5)
            nn.init.normal_(self.image_profile_projection, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection, std=self.text_width ** -0.5)
            nn.init.normal_(self.profile_projection, std=self.profile_width ** -0.5)

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(
            OrderedDict(
                [
                    ("layer1", nn.Linear(in_dim, mlp_dim)),
                    ("bn1", nn.SyncBatchNorm(mlp_dim)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                    ("bn2", nn.SyncBatchNorm(mlp_dim)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("layer3", nn.Linear(mlp_dim, out_dim)),
                ]
            )
        )

    def encode_image(self, eeg: torch.Tensor, pos: torch.Tensor, debug: bool = False) -> torch.Tensor:
        # REVE: [B, C*H, E]
        if debug:
            print(f"Input EEG shape: {eeg.shape}, pos shape: {pos.shape}")
        tokens = self.visual(eeg, pos)
        if debug:
            print(f"REVE output tokens shape: {tokens.shape}")
        # → [B, C, H, E]
        tokens = rearrange(tokens, "b (c h) e -> b c h e", c=self.reve_num_channels)
        # pool the time axis
        if self.time_pool == "mean":
            tokens = tokens.mean(dim=2)
        elif self.time_pool == "max":
            tokens = tokens.max(dim=2).values
        else:
            raise ValueError(f"Unknown time_pool: {self.time_pool}")
        # concat channels → [B, C*E]
        tokens = rearrange(tokens, "b c e -> b (c e)")
        return tokens

    def forward(self, data_dict, debug: bool = False) -> dict:
        eeg = data_dict["eeg"]
        pos = data_dict["pos"]
        report_valid = data_dict["report_valid"]
        text_ids = data_dict["text_ids"]
        text_mask = data_dict["text_mask"]
        demo_token = data_dict["demo_token"]
        disease_token = data_dict["disease_token"]
        medication_token = data_dict["medication_token"]
        disease_mask = data_dict["disease_mask"]
        medication_mask = data_dict["medication_mask"]

        # ---- image ----
        img_embed_raw = self.encode_image(eeg, pos, debug=debug)  # [B, C*E]

        if debug:
            print(f"Raw image embedding shape (pre-projection): {img_embed_raw.shape}")

        if self.vl_projection == "linear":
            img_embed_text = img_embed_raw @ self.image_text_projection
            img_embed_profile = img_embed_raw @ self.image_profile_projection
        else:
            img_embed_text = self.image_text_projection(img_embed_raw)
            img_embed_profile = self.image_profile_projection(img_embed_raw)

        # ---- text ----
        if self.disable_clip or self.disable_text:
            batch_size = img_embed_text.shape[0]
            text_embed = torch.zeros(
                batch_size, self.emb_dim, device=img_embed_text.device, dtype=img_embed_text.dtype
            )
            text_global_feature_pretrained = None
        else:
            text_embed, text_global_feature_pretrained = self.text_encoder(
                input_ids=text_ids, attention_mask=text_mask
            )
            if self.vl_projection == "linear":
                text_embed = text_embed @ self.text_projection
            else:
                text_embed = self.text_projection(text_embed)

        # ---- profile ----
        if self.disable_clip or self.disable_profile:
            batch_size = img_embed_profile.shape[0]
            profile_embed = torch.zeros(
                batch_size,
                self.emb_dim,
                device=img_embed_profile.device,
                dtype=img_embed_profile.dtype,
            )
            one_hot_ehr_vector = None
        else:
            profile_embed, one_hot_ehr_vector = self.profile_encoder(
                demo_token, disease_token, disease_mask, medication_token, medication_mask
            )
            if self.vl_projection == "linear":
                profile_embed = profile_embed @ self.profile_projection
            else:
                profile_embed = self.profile_projection(profile_embed)

        # No reconstruction loss / mask rate when training pure CLIP from a
        # pretrained REVE encoder. Provide tensors that satisfy the CLIPLoss
        # plumbing (mask_rate=0 means every sample passes the mask filter).
        batch_size = img_embed_text.shape[0]
        device = img_embed_text.device
        zero_loss = torch.zeros((), device=device, dtype=img_embed_text.dtype)
        zero_mask_rate = torch.zeros(batch_size, device=device, dtype=img_embed_text.dtype)

        return {
            "image_emb_text": img_embed_text,
            "image_emb_profile": img_embed_profile,
            "img_recon_loss": zero_loss,
            "text_emb": text_embed,
            "profile_emb": profile_embed,
            "logit_scale": self.logit_scale.exp(),
            "logit_scale_profile": self.logit_scale_profile.exp(),
            "report_valid": report_valid,
            "mask_rate": zero_mask_rate,
            "temperatures": None,
            "text_global_feature_pretrained": text_global_feature_pretrained,
            "one_hot_ehr": one_hot_ehr_vector,
        }
