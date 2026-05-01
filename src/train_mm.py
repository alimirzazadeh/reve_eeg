"""Hydra entrypoint for multimodal CLIP training on top of a pretrained REVE encoder."""

import os
import time
from os.path import join as pjoin

import hydra
import torch
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
    get_h5_val_loader,
)
from utils.ddp_setup import ensure_type, get_accelerator, get_logger, save_state
from utils.ema import ModelEMA
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
        pool=clip.reve.pool,
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
        learnable_logit_scale=clip.loss.learnable_logit_scale,
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


def scale_lr(args, world_size: int) -> None:
    """Apply linear LR scaling: lr = blr * batch_size * world_size / 256.
    Skipped when args.trainer.auto_scale_lr is False or absent — the manually
    specified args.trainer.lr is then used as-is.
    """
    if not getattr(args.trainer, "auto_scale_lr", True):
        return
    blr = float(args.trainer.blr)
    bs = int(args.trainer.batch_size)
    accum = int(getattr(args.trainer, "accumulate_grad_batches", 1))
    scaled = blr * bs * max(1, world_size) * accum / 256.0
    print(
        f"Scaling LR: blr={blr} * bs={bs} * world_size={max(1, world_size)} "
        f"* accum={accum} / 256 = {scaled:.3e} (was trainer.lr={args.trainer.lr})"
    )
    args.trainer.lr = scaled


@torch.no_grad()
def run_validation(model, loss_fn, val_loader, accelerator, epoch, prefix="val"):
    """Compute mean CLIP losses over the val set. Returns dict of {prefix}/* metrics."""
    was_training = model.training
    model.eval()
    sum_loss = 0.0
    sum_img_txt = 0.0
    sum_img_profile = 0.0
    sum_txt_valid = 0.0
    sum_profile_valid = 0.0
    n_batches = 0
    for batch in val_loader:
        with accelerator.autocast():
            outputs = model(batch)
            loss_dict = loss_fn(outputs)
        sum_loss += loss_dict["loss"].item()
        sum_img_txt += loss_dict["img_txt_loss"].item()
        sum_img_profile += loss_dict["img_profile_loss"].item()
        sum_txt_valid += loss_dict["img_txt_valid_count"].item()
        sum_profile_valid += loss_dict["img_profile_valid_count"].item()
        n_batches += 1
    if was_training:
        model.train()
    n = max(n_batches, 1)
    return {
        f"{prefix}/loss": sum_loss / n,
        f"{prefix}/img_txt_loss": sum_img_txt / n,
        f"{prefix}/img_profile_loss": sum_img_profile / n,
        f"{prefix}/img_txt_valid_count": sum_txt_valid / n,
        f"{prefix}/img_profile_valid_count": sum_profile_valid / n,
        f"{prefix}/epoch": epoch,
    }


def train_clip(args):
    if args.mode == "debug":
        apply_debug_overrides(args)

    # Build accelerator first so we know the real world size, then scale LR
    # before resolving interpolations (scheduler.peak_lr / end_lr reference
    # ${trainer.lr}).
    init_time = time.time()
    args.checkpointing.state_path = "{:}/{:}".format(args.checkpointing.state_path, args.name)

    accelerator = get_accelerator(args)
    scale_lr(args, world_size=accelerator.num_processes)
    OmegaConf.resolve(args)
    logger.info("Starting multimodal CLIP training")

    model = build_model(args)
    loss_fn = build_loss(args, distributed=accelerator.num_processes > 1)

    if args.mode == "debug" and getattr(args.debug, "max_files", None):
        file_list = _files_from_csv(args.data.split_csv, args.data.h5_dir, "train")
        file_list = file_list[: args.debug.max_files]
        train_loader, len_train = _make_loader_from_list(file_list, args, shuffle=True)
        val_files = _files_from_csv(args.data.split_csv, args.data.h5_dir, "val")
        val_files = val_files[: args.debug.max_files]
        if val_files:
            val_loader, len_val = _make_loader_from_list(val_files, args, shuffle=True)
        else:
            val_loader, len_val = None, 0
    else:
        train_loader, len_train = get_h5_train_loader(args)
        val_loader, len_val = get_h5_val_loader(args)

    # Under multi-GPU, accelerator.prepare wraps the model in DDP, so we
    # cannot ensure_type it back to MultiModalEncoder. Use unwrap_model later
    # if you need to access fields on the underlying module.
    model = accelerator.prepare(model)
    train_loader = ensure_type(accelerator.prepare(train_loader), DataLoader)
    if val_loader is not None:
        val_loader = ensure_type(accelerator.prepare(val_loader), DataLoader)

    ema_cfg = getattr(args.trainer, "ema", None)
    ema = (
        ModelEMA(accelerator.unwrap_model(model), decay=ema_cfg.decay)
        if ema_cfg is not None and ema_cfg.enabled
        else None
    )

    # len(prepared loader) is per-process — needed so the scheduler advances
    # at the right rate (each process calls scheduler.step() once per batch).
    n_iter_per_train = len(train_loader)
    full_run = getattr(args, "scheduler_full_run", False)
    n_iter_for_sched = n_iter_per_train * args.trainer.epochs if full_run else n_iter_per_train


    print("DEBUG: n_iter_per_train", n_iter_per_train, "n_iter_for_sched", n_iter_for_sched)


    n_iter_per_val = len(val_loader) if val_loader is not None else 0
    accelerator.print(
        "Train files:", len_train, "Train batches/process:", n_iter_per_train,
        "Val files:", len_val, "Val batches/process:", n_iter_per_val,
        "N GPUs:", args.trainer.n_gpus,
        "Scheduler steps:", n_iter_for_sched,
    )

    optimizer = get_optimizer(model.parameters(), args.optimizer)
    scheduler = get_lr_scheduler(optimizer, args, n_iter_for_sched)
    optimizer = ensure_type(accelerator.prepare(optimizer), Optimizer)
    scheduler = ensure_type(accelerator.prepare(scheduler), AcceleratedScheduler)
    model.train()

    if args.checkpointing.load_last_state:
        accelerator.load_state(pjoin(args.checkpointing.state_path, "last"))

    debug_mode = args.mode == "debug"
    pbar = tqdm(range(args.trainer.epochs), disable=not accelerator.is_main_process)
    for epoch in pbar:
        start = time.time()
        loss_ema = None
        for batch_idx, batch in enumerate(train_loader):
            # Only print debug shapes on the very first batch to avoid log spam.
            do_debug = debug_mode and epoch == 0 and batch_idx == 0
            with accelerator.accumulate(model), accelerator.autocast():
                optimizer.zero_grad()
                outputs = model(batch, debug=do_debug)
                loss_dict = loss_fn(outputs, debug=do_debug)
                loss = loss_dict["loss"]
                accelerator.backward(loss)
                if args.trainer.grad_clip and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.trainer.grad_clip_norm)
                optimizer.step()

            if ema is not None and accelerator.sync_gradients:
                ema.update(model)
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
                        "lr": optimizer.param_groups[0]["lr"],
                        "img_txt_loss": loss_dict["img_txt_loss"].item(),
                        "img_profile_loss": loss_dict["img_profile_loss"].item(),
                        "img_txt_valid_count": loss_dict["img_txt_valid_count"].item(),
                        "img_profile_valid_count": loss_dict["img_profile_valid_count"].item(),
                    }
                )

        if val_loader is not None:
            val_metrics = run_validation(model, loss_fn, val_loader, accelerator, epoch)
            accelerator.print(
                "Epoch {:3d} val: loss={:.4f} img_txt={:.4f} img_profile={:.4f}".format(
                    epoch,
                    val_metrics["val/loss"],
                    val_metrics["val/img_txt_loss"],
                    val_metrics["val/img_profile_loss"],
                )
            )
            if args.wandb.log:
                accelerator.log(val_metrics)

            if ema is not None:
                val_ema_metrics = run_validation(
                    ema.module, loss_fn, val_loader, accelerator, epoch, prefix="val_ema"
                )
                accelerator.print(
                    "Epoch {:3d} val_ema: loss={:.4f} img_txt={:.4f} img_profile={:.4f}".format(
                        epoch,
                        val_ema_metrics["val_ema/loss"],
                        val_ema_metrics["val_ema/img_txt_loss"],
                        val_ema_metrics["val_ema/img_profile_loss"],
                    )
                )
                if args.wandb.log:
                    accelerator.log(val_ema_metrics)

        save_state(accelerator, args, epoch)

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            epoch_dir = pjoin(args.checkpointing.state_path, f"epoch_{epoch}")
            encoder_path = pjoin(epoch_dir, "encoder.pth")
            model_path = pjoin(epoch_dir, "model_clip.pth")
            accelerator.save(unwrapped.visual.state_dict(), encoder_path)
            accelerator.save(unwrapped.state_dict(), model_path)
            accelerator.print(f"Epoch {epoch}: saved encoder to {encoder_path}")
            accelerator.print(f"Epoch {epoch}: saved full model to {model_path}")

            if ema is not None:
                encoder_ema_path = pjoin(epoch_dir, "encoder_ema.pth")
                model_ema_path = pjoin(epoch_dir, "model_clip_ema.pth")
                accelerator.save(ema.module.visual.state_dict(), encoder_ema_path)
                accelerator.save(ema.state_dict(), model_ema_path)
                accelerator.print(f"Epoch {epoch}: saved EMA encoder to {encoder_ema_path}")
                accelerator.print(f"Epoch {epoch}: saved EMA full model to {model_ema_path}")

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
