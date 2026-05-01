"""Evaluate validation loss across saved multimodal CLIP checkpoints.

Loads `<state_path>/<name>/epoch_{0..N-1}/model_clip.pth` and prints the
mean CLIP losses on the validation set for each epoch.

Default `name=train_mm_old` resolves to:
  /orcd/compute/dinaktbi/001/2026/EEG_FM/reve_clip/checkpoints/states/train_mm_old/epoch_*/model_clip.pth

Override on the CLI, e.g.:
  python src/eval_mm_checkpoints.py name=train_mm_old eval.epochs=20
  python src/eval_mm_checkpoints.py name=train_mm_old eval.use_ema=true
"""

import os
from os.path import join as pjoin

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from configs.resolver import register_resolvers
from train_mm import build_loss, build_model, run_validation
from utils.data_loading_h5_mm import get_h5_val_loader
from utils.ddp_setup import ensure_type, get_accelerator, get_logger


logger = get_logger(__name__)
register_resolvers()


@hydra.main(version_base=None, config_name="config_train_mm", config_path="configs")
def main(args):
    # eval-specific knobs (with defaults so existing config stays untouched)
    eval_cfg = OmegaConf.create(
        {
            "epochs": 20,
            "use_ema": False,
            "checkpoint_dir": None,  # if None, derived from state_path/name below
        }
    )
    if "eval" in args:
        eval_cfg = OmegaConf.merge(eval_cfg, args.eval)

    args.checkpointing.state_path = pjoin(args.checkpointing.state_path, args.name)
    OmegaConf.resolve(args)

    ckpt_dir = eval_cfg.checkpoint_dir or args.checkpointing.state_path
    ckpt_filename = "model_clip_ema.pth" if eval_cfg.use_ema else "model_clip.pth"

    accelerator = get_accelerator(args)
    model = build_model(args)
    loss_fn = build_loss(args, distributed=accelerator.num_processes > 1)

    val_loader, len_val = get_h5_val_loader(args)
    accelerator.print(f"Val files: {len_val}")

    model = accelerator.prepare(model)
    val_loader = ensure_type(accelerator.prepare(val_loader), DataLoader)

    accelerator.print(f"Evaluating checkpoints from {ckpt_dir}")
    accelerator.print(f"Loading file: {ckpt_filename}")

    rows = []
    for epoch in range(eval_cfg.epochs):
        ckpt_path = pjoin(ckpt_dir, f"epoch_{epoch}", ckpt_filename)
        if not os.path.exists(ckpt_path):
            accelerator.print(f"[epoch {epoch}] missing: {ckpt_path} — skipping")
            continue

        state = torch.load(ckpt_path, map_location="cpu")
        # Saved via accelerator.save(unwrapped.state_dict(), ...) in train_mm.py
        accelerator.unwrap_model(model).load_state_dict(state)
        del state

        metrics = run_validation(model, loss_fn, val_loader, accelerator, epoch)
        accelerator.print(
            "Epoch {:3d} val: loss={:.4f} img_txt={:.4f} img_profile={:.4f}  "
            "txt_valid={:.1f} prof_valid={:.1f}".format(
                epoch,
                metrics["val/loss"],
                metrics["val/img_txt_loss"],
                metrics["val/img_profile_loss"],
                metrics["val/img_txt_valid_count"],
                metrics["val/img_profile_valid_count"],
            )
        )
        rows.append((epoch, metrics["val/loss"], metrics["val/img_txt_loss"], metrics["val/img_profile_loss"]))

    if accelerator.is_main_process and rows:
        accelerator.print("\nepoch\tloss\timg_txt\timg_profile")
        for ep, l, lt, lp in rows:
            accelerator.print(f"{ep}\t{l:.4f}\t{lt:.4f}\t{lp:.4f}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
