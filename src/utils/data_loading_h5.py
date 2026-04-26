from __future__ import annotations

"""
H5-based EEG data loading for REVE pretraining on the internal Harvard dataset.

Each epoch, one random window is sampled from each file.
__len__ = number of files — no startup scan needed.

Robustness (mirrors EEG-FM failsafes):
  - known bad files are excluded at construction time
  - SafeDataset wrapper catches per-item exceptions and returns None
  - safe_collate_fn filters None items so bad files never crash training
  - worker_init_fn limits thread counts and trims malloc arenas

Returns items compatible with REVE's MAE training loop:
  (eeg, pos, batch_mask, batch_unmask)
"""

import ctypes
import os
import random
from functools import lru_cache

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.data_loading import create_block_masks

# Known-bad files from EEG-FM repo (bad_h5_files.py)
sys_path_for_bad_files = os.path.join(
    os.path.dirname(__file__), "../../../../EEG_FM/EEG_FM/rep-learning/data"
)
try:
    import sys as _sys
    _sys.path.insert(0, sys_path_for_bad_files)
    from bad_h5_files import bad_h5_files as BAD_H5_FILES
    BAD_H5_FILES = set(BAD_H5_FILES)
except Exception:
    BAD_H5_FILES = set()

CHANNEL_ORDER = [
    "O1", "O2", "T6", "P4", "Pz", "P3", "T5", "T3",
    "C3", "Cz", "C4", "T4", "F8", "F4", "Fz", "F3", "F7", "Fp1", "Fp2",
]
NUM_CHANNELS = len(CHANNEL_ORDER)  # 19


@lru_cache(maxsize=1)
def _load_channel_positions():
    from downstream_tasks.position_utils import load_positions
    return load_positions(electrode_names=CHANNEL_ORDER)  # (19, 3)


class H5EEGDataset(Dataset):
    """
    One random window per file per epoch. No startup scan.

    Known bad files are filtered out at construction time.
    Wrap with SafeDataset + safe_collate_fn to skip any remaining bad items.
    """

    def __init__(
        self,
        file_list,
        data_dir,
        window_duration=2000,
        clip=15.0,
        masking_ratio=0.55,
        masking_window=200,
        masking_overlap=20,
        radius_spat_mask=0.03,
        radius_temp_mask=3,
        dropout_ratio=0.1,
        dropout_radius=0.04,
    ):
        # Filter out known bad files
        filtered = [
            f for f in file_list
            if os.path.basename(f.strip()) not in BAD_H5_FILES
        ]
        if len(filtered) < len(file_list):
            print(f"  Excluded {len(file_list) - len(filtered)} known bad files")

        self.file_list = [os.path.join(data_dir, f.strip()) for f in filtered]
        self.window_duration = window_duration
        self.clip = clip
        self.masking_ratio = masking_ratio
        self.masking_window = masking_window
        self.masking_overlap = masking_overlap
        self.radius_spat_mask = radius_spat_mask
        self.radius_temp_mask = radius_temp_mask
        self.dropout_ratio = dropout_ratio
        self.dropout_radius = dropout_radius

        self._positions_np = _load_channel_positions().numpy()
        self._positions_tensor = torch.from_numpy(self._positions_np).float()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fpath = self.file_list[idx]

        with h5py.File(fpath, "r") as f:
            rec = f["recording"]
            total_samples = rec["data"].shape[0]
            max_start = total_samples - self.window_duration
            start = random.randint(0, max(max_start, 0))
            ch_names_raw = rec["ch_names"][:]
            raw = rec["data"][start : start + self.window_duration, :]  # (T, C_file)

        T, C_file = raw.shape
        if T < self.window_duration:
            pad = np.zeros((self.window_duration - T, C_file), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)

        channels_lower = [
            (n.decode() if isinstance(n, bytes) else n).lower() for n in ch_names_raw
        ]
        eeg = np.zeros((self.window_duration, NUM_CHANNELS), dtype=np.float32)
        for col, ch in enumerate(CHANNEL_ORDER):
            if ch.lower() in channels_lower:
                eeg[:, col] = raw[:, channels_lower.index(ch.lower())]

        mean = eeg.mean(axis=0, keepdims=True)
        std = eeg.std(axis=0, keepdims=True) + 1e-8
        eeg_t = torch.from_numpy((eeg - mean) / std).T.float().clip(-self.clip, self.clip)

        _, h_patches, _ = eeg_t.unfold(1, self.masking_window, self.masking_window - self.masking_overlap).shape
        batch_mask, batch_unmask = create_block_masks(
            NUM_CHANNELS, self.masking_ratio, self.radius_spat_mask, self.radius_temp_mask,
            h_patches, self._positions_np, self.dropout_ratio, self.dropout_radius,
        )

        return eeg_t, self._positions_tensor.clone(), batch_mask, batch_unmask


class SafeDataset(Dataset):
    """Wraps a dataset and returns None for any item that raises an exception."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            return self.dataset[idx]
        except Exception:
            return None


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def safe_collate_fn(batch):
    """Filter None items, collate the rest, and append a skip count."""
    n_skipped = sum(1 for item in batch if item is None)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    try:
        return (*torch.utils.data.dataloader.default_collate(batch), n_skipped)
    except Exception:
        return None


def worker_init_fn(_):
    """Limit thread counts and aggressively return memory from malloc arenas."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 64 * 1024)  # M_MMAP_THRESHOLD = 64KB
        libc.mallopt(-1, 64 * 1024)  # M_TRIM_THRESHOLD = 64KB
    except Exception:
        pass


def _files_from_csv(csv_path, h5_dir, split):
    """Return list of h5 basenames for the given split, matching ProbeLabelHunter logic."""
    import pandas as pd
    labels = pd.read_csv(csv_path)
    patient_ids = {
        'sub-' + str(row['SiteID']) + str(row['BDSPPatientID'])
        for _, row in labels[labels['split'] == split].iterrows()
    }
    files = []
    for fname in sorted(os.listdir(h5_dir)):
        if not fname.endswith('.h5'):
            continue
        if os.path.basename(fname) in BAD_H5_FILES:
            continue
        if fname.split('_')[0] in patient_ids:
            files.append(fname)
    print(f"  CSV split='{split}': {len(files)} files matched from {len(patient_ids)} patient IDs")
    return files


def _make_loader(file_list_path, args, shuffle):
    with open(file_list_path) as fh:
        file_list = [l.strip() for l in fh if l.strip()]

    cfg = args.preprocessing
    mask_cfg = cfg.masking
    dataset = SafeDataset(H5EEGDataset(
        file_list=file_list,
        data_dir=args.data.h5_dir,
        window_duration=cfg.window_duration,
        clip=cfg.clip,
        masking_ratio=mask_cfg.ratio,
        masking_window=mask_cfg.masking_window,
        masking_overlap=mask_cfg.masking_overlap,
        radius_spat_mask=mask_cfg.radius_spat_mask,
        radius_temp_mask=mask_cfg.radius_temp_mask,
        dropout_ratio=mask_cfg.dropout_ratio,
        dropout_radius=mask_cfg.dropout_radius,
    ))
    print(f"  {'Train' if shuffle else 'Val'} dataset: {len(dataset):,} files")

    nw = args.data.loader.num_workers
    loader = DataLoader(
        dataset,
        batch_size=args.trainer.batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        drop_last=shuffle,
        prefetch_factor=args.data.loader.prefetch_factor if nw > 0 else None,
        persistent_workers=False,
        collate_fn=safe_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return loader, len(dataset)


def _make_loader_from_list(file_list, args, shuffle):
    cfg = args.preprocessing
    mask_cfg = cfg.masking
    dataset = SafeDataset(H5EEGDataset(
        file_list=file_list,
        data_dir=args.data.h5_dir,
        window_duration=cfg.window_duration,
        clip=cfg.clip,
        masking_ratio=mask_cfg.ratio,
        masking_window=mask_cfg.masking_window,
        masking_overlap=mask_cfg.masking_overlap,
        radius_spat_mask=mask_cfg.radius_spat_mask,
        radius_temp_mask=mask_cfg.radius_temp_mask,
        dropout_ratio=mask_cfg.dropout_ratio,
        dropout_radius=mask_cfg.dropout_radius,
    ))
    print(f"  {'Train' if shuffle else 'Val'} dataset: {len(dataset):,} files")
    nw = args.data.loader.num_workers
    loader = DataLoader(
        dataset,
        batch_size=args.trainer.batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        drop_last=shuffle,
        prefetch_factor=args.data.loader.prefetch_factor if nw > 0 else None,
        persistent_workers=False,
        collate_fn=safe_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return loader, len(dataset)


def get_h5_train_loader(args):
    split_csv = getattr(args.data, "split_csv", None)
    if split_csv:
        file_list = _files_from_csv(split_csv, args.data.h5_dir, "train")
        return _make_loader_from_list(file_list, args, shuffle=True)
    return _make_loader(args.data.file_list, args, shuffle=True)


def get_h5_val_loader(args):
    split_csv = getattr(args.data, "split_csv", None)
    if split_csv:
        file_list = _files_from_csv(split_csv, args.data.h5_dir, "val")
        if not file_list:
            return None, 0
        return _make_loader_from_list(file_list, args, shuffle=False)
    val_file_list = getattr(args.data, "val_file_list", None)
    if not val_file_list:
        return None, 0
    return _make_loader(val_file_list, args, shuffle=False)
