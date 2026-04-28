"""Hydra entrypoint for multimodal CLIP training on top of a pretrained REVE encoder."""

import os
import time
from os.path import join as pjoin

import hydra
from accelerate.scheduler import AcceleratedScheduler
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.resolver import register_resolvers
from models.loss_clip import CLIPLoss
from models.model_clip import MultiModalEncoder
from utils.data_loading_h5_mm import (
    _files_from_csv,
    _make_loader_from_list,
    get_h5_train_loader,
)
from utils.ddp_setup import ensure_type, get_accelerator, get_logger, save_state
from utils.optim import get_lr_scheduler, get_optimizer


logger = get_logger(__name__)
register_resolvers()


def build_model(args) -> MultiModalEncoder:
    enc = args.encoder
    clip = args.clip
    return MultiModalEncoder(
        # vision (REVE)
        reve_embed_dim=enc.transformer.embed_dim,
        reve_depth=enc.transformer.depth,
        reve_heads=enc.transformer.heads,
        reve_head_dim=enc.transformer.head_dim,
        reve_mlp_dim_ratio=enc.transformer.mlp_dim_ratio,
        reve_use_geglu=enc.transformer.use_geglu,
        reve_freqs=enc.freqs,
        reve_patch_size=enc.patch_size,
        reve_patch_overlap=enc.patch_overlap,
        reve_noise_ratio=enc.noise_ratio,
        reve_num_channels=clip.reve.num_channels,
        reve_checkpoint_path=clip.reve.checkpoint_path,
        freeze_reve=clip.reve.freeze,
        time_pool=clip.reve.time_pool,
        # text
        text_encoder_num_layers=clip.text.num_layers,
        text_encoder_nhead=clip.text.nhead,
        text_model_name=clip.text.model_name,
        text_model_cache_dir=clip.text.cache_dir,
        text_model_context_length=clip.text.context_length,
        use_text_augment=clip.text.use_augment,
        text_aggregation=clip.text.aggregation,
        # profile
        profile_encoder_num_layers=clip.profile.num_layers,
        profile_encoder_nhead=clip.profile.nhead,
        disable_profile_multilevel=clip.profile.disable_multilevel,
        profile_multilevel_embedding_type=clip.profile.multilevel_embedding_type,
        profile_demo_only=clip.profile.demo_only,
        use_profile_augment=clip.profile.use_augment,
        disease_level=clip.profile.disease_level,
        medication_level=clip.profile.medication_level,
        disable_seg_embed=clip.profile.disable_seg_embed,
        # vl
        text_width=clip.text_width,
        profile_width=clip.profile_width,
        vl_projection=clip.vl_projection,
        mlp_dim=clip.mlp_dim,
        emb_dim=clip.emb_dim,
        disable_clip=clip.disable_clip,
        disable_profile=clip.disable_profile,
        disable_text=clip.disable_text,
    )


def build_loss(args, distributed: bool) -> CLIPLoss:
    loss_cfg = args.clip.loss
    return CLIPLoss(
        w1=loss_cfg.w1,  # img_recon (forced 0 because we're not doing MAE)
        w2=loss_cfg.w2,  # img-text
        w3=loss_cfg.w3,  # img-profile
        disable_clip=args.clip.disable_clip,
        disable_profile=args.clip.disable_profile,
        disable_text=args.clip.disable_text,
        mask_rate_max_threshold_clip=loss_cfg.mask_rate_max_threshold_clip,
        mask_rate_min_threshold_recon=loss_cfg.mask_rate_min_threshold_recon,
        distributed=distributed,
        learnable_logit_scale=loss_cfg.learnable_logit_scale,
        soft_contrastive_loss_text=loss_cfg.soft_contrastive_loss_text,
        soft_contrastive_loss_profile=loss_cfg.soft_contrastive_loss_profile,
    )


def apply_debug_overrides(args):
    """Apply args.debug.* overrides in-place. Called only when args.mode == 'debug'."""
    dbg = getattr(args, "debug", None)
    if dbg is None:
        return
    args.trainer.epochs = dbg.epochs
    args.trainer.batch_size = dbg.batch_size
    args.data.loader.num_workers = dbg.num_workers
    args.data.loader.prefetch_factor = dbg.prefetch_factor
    if dbg.disable_wandb:
        args.wandb.log = False
    os.environ["HYDRA_FULL_ERROR"] = "1"
    print("=" * 60)
    print("DEBUG MODE — applied overrides:")
    print(f"  epochs={args.trainer.epochs}, batch_size={args.trainer.batch_size}")
    print(f"  num_workers={args.data.loader.num_workers}, max_files={dbg.max_files}")
    print(f"  wandb.log={args.wandb.log}")
    print("=" * 60)


def train_clip(args):
    if args.mode == "debug":
        apply_debug_overrides(args)
    OmegaConf.resolve(args)
    init_time = time.time()

    args.checkpointing.state_path = "{:}/{:}".format(args.checkpointing.state_path, args.name)

    accelerator = get_accelerator(args)
    logger.info("Starting multimodal CLIP training")

    model = build_model(args)
    loss_fn = build_loss(args, distributed=accelerator.num_processes > 1)

    if args.mode == "debug" and getattr(args.debug, "max_files", None):
        file_list = _files_from_csv(args.data.split_csv, args.data.h5_dir, "train")
        file_list = file_list[: args.debug.max_files]
        train_loader, len_train = _make_loader_from_list(file_list, args, shuffle=True)
    else:
        train_loader, len_train = get_h5_train_loader(args)
    n_iter_per_train = len(train_loader)
    full_run = getattr(args, "scheduler_full_run", False)
    n_iter_for_sched = n_iter_per_train * args.trainer.epochs if full_run else n_iter_per_train
    accelerator.print(
        "Train files:", len_train, "Train batches:", n_iter_per_train,
        "N GPUs:", args.trainer.n_gpus,
        "Scheduler steps:", n_iter_for_sched,
    )

    model = ensure_type(accelerator.prepare(model), MultiModalEncoder)
    optimizer = get_optimizer(model.parameters(), args.optimizer)
    scheduler = get_lr_scheduler(optimizer, args, n_iter_for_sched)

    train_loader = ensure_type(accelerator.prepare(train_loader), DataLoader)
    optimizer = ensure_type(accelerator.prepare(optimizer), Optimizer)
    scheduler = ensure_type(accelerator.prepare(scheduler), AcceleratedScheduler)
    model.train()

    if args.checkpointing.load_last_state:
        accelerator.load_state(pjoin(args.checkpointing.state_path, "last"))

    pbar = tqdm(range(args.trainer.epochs), disable=not accelerator.is_main_process)
    for epoch in pbar:
        start = time.time()
        loss_ema = None
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            with accelerator.accumulate(model), accelerator.autocast():
                optimizer.zero_grad()
                outputs = model(batch)
                loss_dict = loss_fn(outputs)
                loss = loss_dict["loss"]
                accelerator.backward(loss)
                if args.trainer.grad_clip and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.trainer.grad_clip_norm)
                optimizer.step()

            scheduler.step()

            loss_g = loss.item()
            loss_ema = loss_g if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_g
            pbar.set_description(
                "Epoch {:3d} (it. {:3d}/{:3d}) Loss EMA/loss: {:3.3f}/{:3.3f} "
                "(txt {:3.3f} prof {:3.3f}) LR {:.2e}, time {:3.1f}s".format(
                    epoch,
                    batch_idx,
                    n_iter_per_train,
                    loss_ema,
                    loss_g,
                    loss_dict["img_txt_loss"].item(),
                    loss_dict["img_profile_loss"].item(),
                    optimizer.param_groups[0]["lr"],
                    time.time() - start,
                ),
            )

            if args.wandb.log:
                accelerator.log(
                    {
                        "epoch": epoch,
                        "it": batch_idx,
                        "loss": loss_g,
                        "loss_ema": loss_ema,
                        "img_txt_loss": loss_dict["img_txt_loss"].item(),
                        "img_profile_loss": loss_dict["img_profile_loss"].item(),
                        "img_txt_valid_count": loss_dict["img_txt_valid_count"].item(),
                        "img_profile_valid_count": loss_dict["img_profile_valid_count"].item(),
                    }
                )

        save_state(accelerator, args, epoch)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        accelerator.save(unwrapped.visual.state_dict(), "encoder.pth")
        accelerator.save(unwrapped.state_dict(), "model_clip.pth")
        accelerator.print(f"Saved encoder to {os.path.abspath('encoder.pth')}")
        accelerator.print(f"Saved full model to {os.path.abspath('model_clip.pth')}")

    accelerator.print("Training took", time.time() - init_time, "seconds")
    accelerator.end_training()


@hydra.main(version_base=None, config_name="config_train_mm", config_path="configs")
def main(args):
    if args.mode in ["debug", "train"]:
        train_clip(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
