"""
Quick test script for H5EEGDataset (multimodal variant).

Builds a small dataset over a few h5 files, fetches one or more samples, and
prints out the EEG / text / demographics / disease / medication fields so you
can sanity-check the multimodal loading end-to-end.

Run (from repo root):
    PYTHONPATH=src conda run -n reve python src/test_data_loading_h5_mm.py
or with explicit paths:
    PYTHONPATH=src conda run -n reve python src/test_data_loading_h5_mm.py \\
        --h5_dir /orcd/compute/dinaktbi/001/2026/EEG_FM/preprocessed_eeg_v2 \\
        --split_csv /orcd/home/002/alimirz/2026/EEG_FM/EEG_FM/taming-transformers/data/patient_train_val_test_split.csv \\
        --report_dir /orcd/compute/dinaktbi/001/2026/EEG_FM/processed_reports \\
        --num_samples 2
"""

import argparse
import os
import random
import time

from utils.data_loading_h5_mm import (
    H5EEGDataset,
    _files_from_csv,
    DEFAULT_REPORT_DIR,
    DEFAULT_TOKENIZER_CACHE_DIR,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--h5_dir",
        default="/orcd/compute/dinaktbi/001/2026/EEG_FM/preprocessed_eeg_v2",
        help="Directory containing preprocessed .h5 files.",
    )
    p.add_argument(
        "--split_csv",
        default="/orcd/home/002/alimirz/2026/EEG_FM/EEG_FM/taming-transformers/data/patient_train_val_test_split.csv",
        help="CSV with BDSPPatientID/SiteID/split columns. If given, the 'val' split is used.",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Which split to draw files from.",
    )
    p.add_argument(
        "--file_list",
        default=None,
        help="Optional path to a newline-delimited file of h5 basenames (overrides --split_csv).",
    )
    p.add_argument("--report_dir", default=DEFAULT_REPORT_DIR)
    p.add_argument("--tokenizer_cache_dir", default=DEFAULT_TOKENIZER_CACHE_DIR)
    p.add_argument("--window_duration", type=int, default=2000)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_files", type=int, default=200,
                   help="Cap on number of files passed to the dataset (keeps construction fast).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable_clip", action="store_true",
                   help="Skip text/T5 + structured-data loading (returns placeholders).")
    return p.parse_args()


def get_file_list(args):
    if args.file_list is not None:
        with open(args.file_list) as fh:
            files = [l.strip() for l in fh if l.strip()]
    elif args.split_csv and os.path.exists(args.split_csv):
        files = _files_from_csv(args.split_csv, args.h5_dir, args.split)
    else:
        # Fall back to directory listing
        files = sorted(f for f in os.listdir(args.h5_dir) if f.endswith(".h5"))
        print(f"  Directory scan: found {len(files)} .h5 files in {args.h5_dir}")
    if args.max_files and len(files) > args.max_files:
        random.Random(args.seed).shuffle(files)
        files = files[: args.max_files]
        print(f"  Capped file list to {len(files)} files (--max_files={args.max_files})")
    return files


def print_sample(sample, idx):
    print(f"========== Sample {idx} ==========", flush=True)
    print(f"EEG file: {sample['eeg_file']}", flush=True)
    print(f"EEG shape: {tuple(sample['eeg'].shape)}", flush=True)
    print(f"Pos shape: {tuple(sample['pos'].shape)}", flush=True)
    print(f"batch_mask shape: {tuple(sample['batch_mask'].shape)}", flush=True)
    print(f"batch_unmask shape: {tuple(sample['batch_unmask'].shape)}", flush=True)

    print(f"Report valid: {sample['report_valid']}", flush=True)
    print(f"Text IDs shape: {tuple(sample['text_ids'].shape)}", flush=True)
    print(f"Text IDs: {sample['text_ids'].tolist()}", flush=True)
    print(f"Text mask shape: {tuple(sample['text_mask'].shape)}", flush=True)
    print(f"Text mask sum (real tokens): {sample['text_mask'].sum().item()}", flush=True)

    print(f"Demo token shape: {tuple(sample['demo_token'].shape)}", flush=True)
    print(f"Demo token values (age, sex, race): {sample['demo_token'].tolist()}", flush=True)

    print(f"Disease token shape: {tuple(sample['disease_token'].shape)}", flush=True)
    for level in range(sample['disease_token'].shape[0]):
        print(f"  Disease token level {level} shape: {tuple(sample['disease_token'][level].shape)}", flush=True)
        print(f"  Disease token level {level} values: {sample['disease_token'][level].tolist()}", flush=True)
    print(f"Disease mask shape: {tuple(sample['disease_mask'].shape)}", flush=True)
    print(f"Disease mask sum (num diseases): {sample['disease_mask'].sum().item()}", flush=True)

    print(f"Medication token shape: {tuple(sample['medication_token'].shape)}", flush=True)
    for level in range(sample['medication_token'].shape[0]):
        print(f"  Medication token level {level} shape: {tuple(sample['medication_token'][level].shape)}", flush=True)
        print(f"  Medication token level {level} values: {sample['medication_token'][level].tolist()}", flush=True)
    print(f"Medication mask shape: {tuple(sample['medication_mask'].shape)}", flush=True)
    print(f"Medication mask sum (num medications): {sample['medication_mask'].sum().item()}", flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)

    file_list = get_file_list(args)
    if not file_list:
        raise SystemExit("No files to load — check --h5_dir / --split_csv / --file_list.")

    dataset = H5EEGDataset(
        file_list=file_list,
        data_dir=args.h5_dir,
        window_duration=args.window_duration,
        fixed_start=False,
        disable_clip=args.disable_clip,
        report_dir=args.report_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        mode='val',
    )
    print(f"Dataset length: {len(dataset)}", flush=True)
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty after filtering bad files.")

    n = min(args.num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), n)

    for idx in sample_indices:
        t0 = time.time()
        sample = dataset[idx]
        dt = time.time() - t0
        print(f"[idx={idx}] __getitem__ took {dt:.3f}s", flush=True)
        print_sample(sample, idx)


if __name__ == "__main__":
    main()
