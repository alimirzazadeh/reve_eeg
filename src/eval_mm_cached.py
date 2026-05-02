"""Evaluate CLIP loss on cached image embeddings.

Skips the REVE visual encoder by loading per-file cached embeddings from
`<embeddings_dir>/<basename>.pt`, where `<basename>` matches the `.h5` file
without the `.h5` suffix. Each cached embedding is paired with text /
demographics / disease / medication tokens parsed by the same code path
used during training (H5EEGDataset), then fed through the rest of
MultiModalEncoder (projections + text/profile encoders) to compute the
CLIP loss.

Expected cached tensor shape per file: `[1, vision_width]` (squeezed to
`[vision_width]` before collation).

Usage:
  python src/eval_mm_cached.py \
    eval.model_clip_path=/path/to/model_clip.pth \
    eval.embeddings_dir=/path/to/cached_embeddings \
    [eval.split=val] [eval.batch_size=64]
"""

import os
from os.path import join as pjoin

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from configs.resolver import register_resolvers
from models.model_clip import MultiModalEncoder
from train_mm import build_loss, build_model, run_validation
from utils.data_loading_h5_mm import (
    H5EEGDataset,
    _files_from_csv,
    worker_init_fn,
)
from utils.ddp_setup import ensure_type, get_accelerator, get_logger


logger = get_logger(__name__)
register_resolvers()


class CachedMultiModalEncoder(MultiModalEncoder):
    """Same module as MultiModalEncoder, but `forward` consumes a cached
    image embedding (`batch['img_embed']`) instead of running the REVE
    visual encoder. This way `run_validation(model, loss_fn, loader, ...)`
    works unchanged.
    """

    def forward(self, data_dict, debug: bool = False) -> dict:
        img_embed_raw = data_dict["img_embed"]  # [B, vision_width]
        if img_embed_raw.shape[-1] != self.vision_width:
            raise ValueError(
                f"Cached embedding last-dim {img_embed_raw.shape[-1]} != "
                f"model.vision_width {self.vision_width}. Check pool / encoder match."
            )

        if self.vl_projection == "linear":
            img_embed_text = img_embed_raw @ self.image_text_projection
            img_embed_profile = img_embed_raw @ self.image_profile_projection
        else:
            img_embed_text = self.image_text_projection(img_embed_raw)
            img_embed_profile = self.image_profile_projection(img_embed_raw)

        if self.disable_clip or self.disable_text:
            bs = img_embed_text.shape[0]
            text_embed = torch.zeros(
                bs, self.emb_dim, device=img_embed_text.device, dtype=img_embed_text.dtype
            )
            text_global_feature_pretrained = None
        else:
            text_embed, text_global_feature_pretrained = self.text_encoder(
                input_ids=data_dict["text_ids"], attention_mask=data_dict["text_mask"]
            )
            if self.vl_projection == "linear":
                text_embed = text_embed @ self.text_projection
            else:
                text_embed = self.text_projection(text_embed)

        if self.disable_clip or self.disable_profile:
            bs = img_embed_profile.shape[0]
            profile_embed = torch.zeros(
                bs, self.emb_dim, device=img_embed_profile.device, dtype=img_embed_profile.dtype
            )
            one_hot_ehr_vector = None
        else:
            profile_embed, one_hot_ehr_vector = self.profile_encoder(
                data_dict["demo_token"],
                data_dict["disease_token"],
                data_dict["disease_mask"],
                data_dict["medication_token"],
                data_dict["medication_mask"],
            )
            if self.vl_projection == "linear":
                profile_embed = profile_embed @ self.profile_projection
            else:
                profile_embed = self.profile_projection(profile_embed)

        bs = img_embed_text.shape[0]
        device = img_embed_text.device
        zero_loss = torch.zeros((), device=device, dtype=img_embed_text.dtype)
        zero_mask_rate = torch.zeros(bs, device=device, dtype=img_embed_text.dtype)

        return {
            "image_emb_text": img_embed_text,
            "image_emb_profile": img_embed_profile,
            "img_recon_loss": zero_loss,
            "text_emb": text_embed,
            "profile_emb": profile_embed,
            "logit_scale": self.logit_scale.exp(),
            "logit_scale_profile": self.logit_scale_profile.exp(),
            "report_valid": data_dict["report_valid"],
            "mask_rate": zero_mask_rate,
            "temperatures": None,
            "text_global_feature_pretrained": text_global_feature_pretrained,
            "one_hot_ehr": one_hot_ehr_vector,
        }


class CachedEmbeddingDataset(H5EEGDataset):
    """Reuses H5EEGDataset's metadata parsing, but loads a cached image
    embedding from `<embeddings_dir>/<basename>.pt` instead of reading the
    h5 EEG window. Skips the h5-validation pass at __init__ time.
    """

    def __init__(self, file_list, embeddings_dir, **kwargs):
        # Bypass parent's h5 validation by passing an empty list, then set
        # the real file basenames after init. data_dir is irrelevant here —
        # we never read it — so we point it at embeddings_dir for clarity.
        super().__init__(file_list=[], data_dir=embeddings_dir, **kwargs)
        self.file_list = list(file_list)
        self.embeddings_dir = embeddings_dir

    def _load_cached_embedding(self, eeg_file):
        stem = eeg_file[:-3] if eeg_file.endswith(".h5") else eeg_file
        pt_path = pjoin(self.embeddings_dir, stem + ".pt")
        emb = torch.load(pt_path, map_location="cpu", weights_only=True).float()
        if emb.dim() != 2 or emb.shape[0] != 1:
            raise ValueError(
                f"{pt_path}: expected shape [1, vision_width], got {tuple(emb.shape)}"
            )
        return emb.squeeze(0)

    def __getitem__(self, idx):
        eeg_file = self.file_list[idx]

        img_embed = self._load_cached_embedding(eeg_file)

        cleaned_report, report_valid = self._load_and_clean_report(eeg_file)
        text_ids, text_mask = self._tokenize_text(cleaned_report)
        age_token, sex_token, race_token = self._load_demo_data(eeg_file)
        demo_token = torch.tensor([age_token, sex_token, race_token], dtype=torch.long)
        disease_token, disease_mask = self._load_disease_data(eeg_file)
        medication_token, medication_mask = self._load_medication_data(eeg_file)

        return {
            "img_embed": img_embed,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "demo_token": demo_token,
            "disease_token": disease_token,
            "medication_token": medication_token,
            "disease_mask": disease_mask,
            "medication_mask": medication_mask,
            "report_valid": report_valid,
            "eeg_file": eeg_file,
        }


def _make_cached_loader(file_list, embeddings_dir, args, batch_size):
    cfg = args.preprocessing
    mask_cfg = cfg.masking
    dataset = CachedEmbeddingDataset(
        file_list=file_list,
        embeddings_dir=embeddings_dir,
        window_duration=cfg.window_duration,
        clip=cfg.clip,
        masking_ratio=mask_cfg.ratio,
        masking_window=mask_cfg.masking_window,
        masking_overlap=mask_cfg.masking_overlap,
        radius_spat_mask=mask_cfg.radius_spat_mask,
        radius_temp_mask=mask_cfg.radius_temp_mask,
        dropout_ratio=mask_cfg.dropout_ratio,
        dropout_radius=mask_cfg.dropout_radius,
        fixed_start=True,
        disable_clip=getattr(args.data, "disable_clip", False),
        report_dir=getattr(
            args.data, "report_dir", "/orcd/compute/dinaktbi/001/2026/EEG_FM/processed_reports"
        ),
        tokenizer_model_name=getattr(args.data, "tokenizer_model_name", "google/t5-v1_1-base"),
        tokenizer_cache_dir=getattr(
            args.data, "tokenizer_cache_dir", "/orcd/data/dinaktbi/001/2026/EEG_FM/model_weights"
        ),
        max_text_length=getattr(args.data, "max_text_length", 256),
        max_num_disease=getattr(args.data, "max_num_disease", 30),
        max_num_medication=getattr(args.data, "max_num_medication", 50),
        num_chunks_per_sample=getattr(args.data, "num_chunks_per_sample", 1),
        mode="val",
    )
    print(f"  Cached-embedding dataset: {len(dataset):,} files")

    nw = args.data.loader.num_workers
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.data.loader.prefetch_factor if nw > 0 else None,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )
    return loader, len(dataset)


def build_cached_model(args) -> CachedMultiModalEncoder:
    """Same as train_mm.build_model but instantiates CachedMultiModalEncoder."""
    base = build_model(args)
    cached = CachedMultiModalEncoder.__new__(CachedMultiModalEncoder)
    cached.__dict__.update(base.__dict__)
    return cached


@hydra.main(version_base=None, config_name="config_train_mm", config_path="configs")
def main(args):
    eval_cfg = OmegaConf.create(
        {
            "model_clip_path": None,
            "embeddings_dir": None,
            "split": "val",
            "batch_size": None,
        }
    )
    if "eval" in args:
        eval_cfg = OmegaConf.merge(eval_cfg, args.eval)

    if eval_cfg.model_clip_path is None:
        raise ValueError("eval.model_clip_path must be set (path to model_clip.pth)")
    if eval_cfg.embeddings_dir is None:
        raise ValueError("eval.embeddings_dir must be set (dir of cached .pt files)")

    OmegaConf.resolve(args)

    accelerator = get_accelerator(args)
    model = build_cached_model(args)
    loss_fn = build_loss(args, distributed=accelerator.num_processes > 1)

    accelerator.print(f"Loading model_clip from {eval_cfg.model_clip_path}")
    state = torch.load(eval_cfg.model_clip_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    accelerator.print(
        f"  loaded (missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if missing:
        accelerator.print(f"  first missing: {missing[:5]}")
    if unexpected:
        accelerator.print(f"  first unexpected: {unexpected[:5]}")
    del state

    file_list = _files_from_csv(args.data.split_csv, args.data.h5_dir, eval_cfg.split)
    kept = [
        f for f in file_list
        if os.path.exists(pjoin(eval_cfg.embeddings_dir, f[:-3] + ".pt"))
    ]
    if len(kept) < len(file_list):
        accelerator.print(
            f"  Dropping {len(file_list) - len(kept)} files without cached embedding "
            f"(kept {len(kept)}/{len(file_list)})"
        )
    file_list = kept

    bs = eval_cfg.batch_size or args.trainer.batch_size
    loader, n = _make_cached_loader(file_list, eval_cfg.embeddings_dir, args, bs)
    accelerator.print(f"Eval files: {n}, batch_size: {bs}")

    model = accelerator.prepare(model)
    loader = ensure_type(accelerator.prepare(loader), DataLoader)

    metrics = run_validation(model, loss_fn, loader, accelerator, epoch=0)
    accelerator.print(
        "CLIP eval (cached): loss={:.4f}  img_txt={:.4f}  img_profile={:.4f}  "
        "txt_valid={:.1f}  prof_valid={:.1f}".format(
            metrics["val/loss"],
            metrics["val/img_txt_loss"],
            metrics["val/img_profile_loss"],
            metrics["val/img_txt_valid_count"],
            metrics["val/img_profile_valid_count"],
        )
    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
