"""
REVE pretraining on the internal Harvard EEG dataset stored as h5 files.

Identical training loop to train.py, but uses H5EEGDataset / get_h5_train_loader
instead of the original LMDB/HDF5 foundation-data loader.

Also adds:
  - TensorBoard logging (train loss, val loss, LR)
  - Validation loss computed each epoch
  - Dataset size printed at startup

Launch (single GPU):
    cd /home/alimirz/2026/reve_eeg
    PYTHONPATH=src conda run -n reve python src/pretrain_h5.py

Launch (multi-GPU with accelerate):
    cd /home/alimirz/2026/reve_eeg
    PYTHONPATH=src conda run -n reve accelerate launch src/pretrain_h5.py

Hydra config: src/configs/config_pretrain_h5.yaml

TensorBoard:
    tensorboard --logdir <checkpointing.path>/tb_logs
"""

import os
import time
from os.path import join as pjoin

import hydra
import torch
from accelerate.scheduler import AcceleratedScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.resolver import register_resolvers
from configs.validate import validate_train
from models.mae import MAE
from utils.data_loading_h5 import get_h5_train_loader, get_h5_val_loader
from utils.ddp_setup import ensure_type, get_accelerator, get_logger, save_state, save_encoder
from utils.optim import get_lr_scheduler, get_optimizer



logger = get_logger(__name__)
register_resolvers()


def _run_val_epoch(mae, val_loader, accelerator):
    """Compute mean reconstruction loss on the validation set."""
    mae.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            x, pos, b_m, b_u, _ = batch
            with accelerator.autocast():
                loss = mae(x, pos, b_m, b_u)
            total_loss += loss.item()
            n_batches += 1
    mae.train()
    return total_loss / max(n_batches, 1)


def train_h5(args):
    validate_train(args)
    init_time = time.time()

    args.checkpointing.state_path = "{:}/{:}".format(args.checkpointing.state_path, args.name)

    accelerator = get_accelerator(args)
    logger.info("Starting h5 pretraining")

    mae = MAE(args)

    # -------------------------------------------------------------------------
    # Build data loaders and print dataset sizes
    # -------------------------------------------------------------------------
    train_loader, n_train = get_h5_train_loader(args)
    val_loader, n_val = get_h5_val_loader(args)

    accelerator.print("=" * 60)
    accelerator.print(f"  Train files : {n_train:,}  (1 random window/file/epoch)")
    accelerator.print(f"  Val files   : {n_val:,}" if n_val else "  Val files   : (none)")
    accelerator.print("=" * 60)

    # Iterations per epoch per process (scheduler needs this before prepare())
    n_iter_per_train = n_train // (
        args.trainer.batch_size * args.trainer.n_gpus * args.trainer.n_nodes
    )
    accelerator.print(
        f"N_GPUS: {args.trainer.n_gpus}  N_NODES: {args.trainer.n_nodes}  "
        f"batch_size: {args.trainer.batch_size}  iters/epoch: {n_iter_per_train}"
    )

    # -------------------------------------------------------------------------
    # TensorBoard — only on main process, logs to checkpointing.path/tb_logs
    # -------------------------------------------------------------------------
    tb_log_dir = pjoin(
        args.checkpointing.path, "tb_logs",
        f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    if accelerator.is_main_process:
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        accelerator.print(f"TensorBoard logs → {tb_log_dir}")
        accelerator.print(f"  tensorboard --logdir {tb_log_dir}")
    else:
        writer = None

    # -------------------------------------------------------------------------
    # Prepare model, optimizer, scheduler, loaders
    # -------------------------------------------------------------------------
    mae = accelerator.prepare(mae)

    optimizer = get_optimizer(mae.parameters(), args.optimizer)
    n_total_steps = n_iter_per_train * args.trainer.epochs
    scheduler = get_lr_scheduler(optimizer, args, n_total_steps)

    train_loader = ensure_type(accelerator.prepare(train_loader), DataLoader)
    if val_loader is not None:
        val_loader = ensure_type(accelerator.prepare(val_loader), DataLoader)
    optimizer = ensure_type(accelerator.prepare(optimizer), Optimizer)
    scheduler = ensure_type(accelerator.prepare(scheduler), AcceleratedScheduler)
    mae.train()

    if args.checkpointing.load_last_state:
        accelerator.load_state(pjoin(args.checkpointing.state_path, "last"))

    global_step = 0
    pbar = tqdm(range(args.trainer.epochs))

    for epoch in pbar:
        start = time.time()
        loss_ema = None
        epoch_losses = []
        epoch_skipped = 0

        # -- Train loop -------------------------------------------------------
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            x, pos, b_m, b_u, n_skipped = batch
            epoch_skipped += n_skipped
            with accelerator.accumulate(mae), accelerator.autocast():
                optimizer.zero_grad()
                loss = mae(x, pos, b_m, b_u)
                accelerator.backward(loss)
                if args.trainer.grad_clip and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(mae.parameters(), args.trainer.grad_clip_norm)
                optimizer.step()

            loss_g = loss.item()
            loss_ema = loss_g if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_g
            epoch_losses.append(loss_g)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            global_step += 1

            # Log every step to TensorBoard
            if writer is not None:
                writer.add_scalar("train/loss", loss_g, global_step)
                writer.add_scalar("train/loss_ema", loss_ema, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)

            if args.wandb.log:
                accelerator.log(
                    {"epoch": epoch, "it": batch_idx, "loss_ema": loss_ema, "loss": loss_g}
                )

            pbar.set_description(
                "Epoch {:3d} ({:4d}/{:4d}) loss {:.4f} ema {:.4f} lr {:.2e} {:.1f}s".format(
                    epoch, batch_idx, n_iter_per_train,
                    loss_g, loss_ema, current_lr,
                    time.time() - start,
                )
            )

        # -- Epoch-level train summary ----------------------------------------
        epoch_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        skip_pct = 100.0 * epoch_skipped / max(n_train, 1)
        if writer is not None:
            writer.add_scalar("epoch/train_loss", epoch_train_loss, epoch)

        # -- Validation (every other epoch) -----------------------------------
        val_loss = None
        if val_loader is not None and epoch % 2 == 0:
            val_loss = _run_val_epoch(mae, val_loader, accelerator)
            if writer is not None:
                writer.add_scalar("epoch/val_loss", val_loss, epoch)
            accelerator.print(
                f"\nEpoch {epoch:3d} | train_loss {epoch_train_loss:.4f} "
                f"| val_loss {val_loss:.4f} | skipped {epoch_skipped} ({skip_pct:.1f}%) "
                f"| {time.time()-start:.1f}s"
            )
        elif val_loader is None and epoch == 0:
            accelerator.print("  WARNING: no validation files found — check split_csv or val_file_list")
            accelerator.print(
                f"\nEpoch {epoch:3d} | train_loss {epoch_train_loss:.4f} "
                f"| skipped {epoch_skipped} ({skip_pct:.1f}%) | {time.time()-start:.1f}s"
            )
        else:
            accelerator.print(
                f"\nEpoch {epoch:3d} | train_loss {epoch_train_loss:.4f} "
                f"| skipped {epoch_skipped} ({skip_pct:.1f}%) | {time.time()-start:.1f}s"
            )

        if writer is not None:
            writer.flush()

        # -- Checkpointing ----------------------------------------------------
        save_state(accelerator, args, epoch)
        save_encoder(accelerator, mae, args, epoch)

    # -------------------------------------------------------------------------
    # Final encoder save
    # -------------------------------------------------------------------------
    final_encoder_path = pjoin(args.checkpointing.state_path, "encoder_final.pth")
    encoder = accelerator.unwrap_model(mae).encoder
    accelerator.save(encoder.state_dict(), final_encoder_path)
    accelerator.print(f"Final encoder saved → {os.path.abspath(final_encoder_path)}")

    if writer is not None:
        writer.close()

    accelerator.print(f"Training took {time.time() - init_time:.1f}s")
    accelerator.end_training()


@hydra.main(version_base=None, config_name="config_pretrain_h5", config_path="configs")
def main(args):
    if args.mode in ["debug", "train"]:
        train_h5(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
