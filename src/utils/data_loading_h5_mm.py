from __future__ import annotations

"""
H5-based EEG data loading for REVE pretraining on the internal Harvard dataset.

Multimodal variant: in addition to the EEG window + masks, each item also
returns tokenized text, demographics, diseases, and medications parsed from
the EEG filename (mirroring EEGTextDataset).

Each epoch, one random window is sampled from each file.
__len__ = number of files — no startup scan needed.
"""

import ctypes
import json
import os
import random
import sys
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.data_loading import create_block_masks
from utils.report_cleaner_v1 import clean_summary

# Known-bad files from EEG-FM repo (bad_h5_files.py)
sys_path_for_bad_files = os.path.join(
    os.path.dirname(__file__), "../../../../EEG_FM/EEG_FM/rep-learning/data"
)
try:
    sys.path.insert(0, sys_path_for_bad_files)
    from bad_h5_files import bad_h5_files as BAD_H5_FILES
    BAD_H5_FILES = set(BAD_H5_FILES)
except Exception:
    BAD_H5_FILES = set()

CHANNEL_ORDER = [
    "O1", "O2", "T6", "P4", "Pz", "P3", "T5", "T3",
    "C3", "Cz", "C4", "T4", "F8", "F4", "Fz", "F3", "F7", "Fp1", "Fp2",
]
NUM_CHANNELS = len(CHANNEL_ORDER)  # 19


# ---------------------------------------------------------------------------
# Structured-data assets (loaded once at import time)
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.dirname(__file__)
METADATA_CSV_PATH = '/orcd/data/dinaktbi/001/2026/EEG_FM/HEEDB_Metadata'
DEFAULT_REPORT_DIR = '/orcd/compute/dinaktbi/001/2026/EEG_FM/processed_reports'
DEFAULT_TOKENIZER_CACHE_DIR = '/orcd/data/dinaktbi/001/2026/EEG_FM/model_weights'

df_demo = pd.read_csv(os.path.join(_DATA_DIR, 'HEEDB_patients.csv'), dtype=str)
df_icd10 = pd.read_csv(os.path.join(_DATA_DIR, 'HEEDB_ICD10_for_Neurology.csv'), dtype=str)
df_atc = pd.read_csv(os.path.join(_DATA_DIR, 'HEEDB_Medication_ATC.csv'), dtype=str)

# Per-site session-level metadata (only present on the cluster)
df_metadata = {}
for _site in ['S0001', 'S0002', 'I0002', 'I0003']:
    _path = os.path.join(METADATA_CSV_PATH, f'{_site}_EEG__reports_findings.csv')
    try:
        df_metadata[_site] = pd.read_csv(_path, dtype=str)
    except FileNotFoundError:
        pass

with open(os.path.join(_DATA_DIR, 'disease_dict.json'), 'r') as _f:
    DISEASE_MULTI_LEVEL_DICT = json.load(_f)
with open(os.path.join(_DATA_DIR, 'medication_dict.json'), 'r') as _f:
    MEDICATION_MULTI_LEVEL_DICT = json.load(_f)

SEX_DICT = {'F': 0, 'M': 1, 'Unknown': 2}
RACE_DICT = {
    'American Indian or Alaska Native': 0,
    'Asian': 1,
    'Black or African American': 2,
    'Multiracial': 3,
    'Native Hawaiian or Other Pacific Islander': 4,
    'Other Race': 5,
    'White': 6,
    'Unavailable': 7,
}
NUM_AGE_GROUP = 10  # NUM_AGE_GROUP itself is reserved as the N/A bucket


def get_age_token(age):
    """Bucket age into [0, NUM_AGE_GROUP-1]; NUM_AGE_GROUP reserved for N/A."""
    clamped_age = min(max(age, 0), 100)
    bucket_size = 100 / NUM_AGE_GROUP
    index = int(clamped_age / bucket_size)
    return min(index, NUM_AGE_GROUP - 1)


@lru_cache(maxsize=1)
def _load_channel_positions():
    from downstream_tasks.position_utils import load_positions
    return load_positions(electrode_names=CHANNEL_ORDER)  # (19, 3)


class H5EEGDataset(Dataset):
    """
    One random window per file per epoch. No startup scan.

    Returns a dict:
      eeg, pos, batch_mask, batch_unmask,
      text_ids, text_mask, demo_token,
      disease_token, medication_token, disease_mask, medication_mask,
      report_valid, eeg_file
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
        fixed_start=False,
        # multimodal args
        disable_clip=False,
        report_dir=DEFAULT_REPORT_DIR,
        tokenizer_model_name="google/t5-v1_1-base",
        tokenizer_cache_dir=DEFAULT_TOKENIZER_CACHE_DIR,
        max_text_length=256,
        max_num_disease=30,
        max_num_medication=50,
        mode='train',
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
        self.fixed_start = fixed_start
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

        # multimodal config
        self.disable_clip = disable_clip
        self.report_dir = report_dir
        self.max_text_length = max_text_length
        self.max_num_disease = max_num_disease
        self.max_num_medication = max_num_medication
        self.mode = mode

        if not self.disable_clip:
            from transformers import T5Tokenizer
            print(f"Loading T5 tokenizer: {tokenizer_model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(
                tokenizer_model_name, cache_dir=tokenizer_cache_dir
            )
        else:
            self.tokenizer = None

    def __len__(self):
        return len(self.file_list)

    # ------------------------------------------------------------------
    # Filename parsing + structured data lookups
    # ------------------------------------------------------------------

    def _get_patient_info(self, eeg_file):
        patient_info = eeg_file.split('_')[0].split('-')[1]
        site_id = patient_info[:5]
        patient_id = patient_info[5:]
        try:
            session_id = eeg_file.split('_ses-')[1].split('_')[0]
        except IndexError:
            session_id = None
        return site_id, patient_id, session_id

    def _load_demo_data(self, eeg_file):
        site_id, patient_id, session_id = self._get_patient_info(eeg_file)

        age_token = NUM_AGE_GROUP  # default to N/A
        if site_id in df_metadata and session_id is not None:
            df = df_metadata[site_id]
            row = df[(df['BDSPPatientID'].astype(str) == str(patient_id))
                     & (df['SessionID'].astype(str) == str(session_id))]
            if len(row) == 1:
                try:
                    age_token = get_age_token(float(row['AgeAtVisit'].item()))
                except Exception:
                    age_token = NUM_AGE_GROUP

        row = df_demo[(df_demo['SiteID'] == site_id) & (df_demo['BDSPPatientID'] == patient_id)]
        if len(row) == 1:
            sex_token = SEX_DICT.get(row['Sex'].item(), SEX_DICT['Unknown'])
            race_token = RACE_DICT.get(row['RaceAndEthnicity'].item(), RACE_DICT['Unavailable'])
        else:
            sex_token = SEX_DICT['Unknown']
            race_token = RACE_DICT['Unavailable']
        return age_token, sex_token, race_token

    def _load_disease_data(self, eeg_file):
        site_id, patient_id, _ = self._get_patient_info(eeg_file)
        row = df_icd10[(df_icd10['SiteID'] == site_id) & (df_icd10['BDSPPatientID'] == patient_id)]
        valid_codes = set(DISEASE_MULTI_LEVEL_DICT['L1'].keys())
        if len(row) == 1:
            icd_columns = df_icd10.columns[5:]
            raw_text = [str(x) for x in row[icd_columns].values[0] if pd.notna(x)]
            code_list = " ".join(raw_text).split()
            code_list = list({c for c in code_list if c in valid_codes})
            if self.mode == 'train':
                random.shuffle(code_list)
        else:
            code_list = []

        num_levels = len(DISEASE_MULTI_LEVEL_DICT)
        disease_token = torch.zeros(num_levels, self.max_num_disease, dtype=torch.long)
        disease_mask = torch.zeros(self.max_num_disease, dtype=torch.long)
        for level in range(num_levels):
            level_dict = DISEASE_MULTI_LEVEL_DICT[f'L{level + 1}']
            disease_token[level] = len(level_dict)  # pad index
            for i, code in enumerate(code_list[:self.max_num_disease]):
                disease_token[level, i] = level_dict[code]
                disease_mask[i] = 1
        return disease_token, disease_mask

    def _load_medication_data(self, eeg_file):
        site_id, patient_id, _ = self._get_patient_info(eeg_file)
        row = df_atc[(df_atc['SiteID'] == site_id) & (df_atc['BDSPPatientID'] == patient_id)]
        valid_codes = set(MEDICATION_MULTI_LEVEL_DICT['L1'].keys())
        if len(row) == 1:
            medication_columns = [
                'Nervous System Drugs',
                'Antineoplastic And Immunomodulating Agents',
                'Systemic Hormonal Preparations, Excl. Sex Hormones And Insulins',
            ]
            raw_text = [str(x) for x in row[medication_columns].values[0] if pd.notna(x)]
            code_list = " ".join(raw_text).split()
            code_list = list({c for c in code_list if c in valid_codes})
            if self.mode == 'train':
                random.shuffle(code_list)
        else:
            code_list = []

        num_levels = len(MEDICATION_MULTI_LEVEL_DICT)
        medication_token = torch.zeros(num_levels, self.max_num_medication, dtype=torch.long)
        medication_mask = torch.zeros(self.max_num_medication, dtype=torch.long)
        for level in range(num_levels):
            level_dict = MEDICATION_MULTI_LEVEL_DICT[f'L{level + 1}']
            medication_token[level] = len(level_dict)  # pad index
            for i, code in enumerate(code_list[:self.max_num_medication]):
                medication_token[level, i] = level_dict[code]
                medication_mask[i] = 1
        return medication_token, medication_mask

    def _find_report_file(self, eeg_file):
        prefix = eeg_file.split('preprocessed-eeg')[0]
        from pathlib import Path
        report_files = list(Path(self.report_dir).glob(f'{prefix}*'))
        if len(report_files) == 1:
            return report_files[0]
        return None

    def _load_and_clean_report(self, eeg_file):
        report_path = self._find_report_file(eeg_file)
        if report_path is None:
            return "", False
        try:
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as fh:
                report_text = fh.read()
        except FileNotFoundError:
            return "", False

        cleaned_text = clean_summary(report_text)
        if not cleaned_text or len(cleaned_text.strip()) == 0:
            return "", False
        return cleaned_text, True

    def _tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
        )
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        fpath = self.file_list[idx]
        eeg_file = os.path.basename(fpath)

        with h5py.File(fpath, "r") as f:
            rec = f["recording"]
            total_samples = rec["data"].shape[0]
            max_start = total_samples - self.window_duration
            start = 0 if self.fixed_start else random.randint(0, max(max_start, 0))
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

        # Multimodal fields
        if self.disable_clip:
            text_ids = torch.zeros(self.max_text_length, dtype=torch.long)
            text_mask = torch.zeros(self.max_text_length, dtype=torch.long)
            demo_token = torch.zeros(3, dtype=torch.long)
            disease_token = torch.zeros(
                len(DISEASE_MULTI_LEVEL_DICT), self.max_num_disease, dtype=torch.long
            )
            medication_token = torch.zeros(
                len(MEDICATION_MULTI_LEVEL_DICT), self.max_num_medication, dtype=torch.long
            )
            disease_mask = torch.zeros(self.max_num_disease, dtype=torch.long)
            medication_mask = torch.zeros(self.max_num_medication, dtype=torch.long)
            report_valid = False
        else:
            cleaned_report, report_valid = self._load_and_clean_report(eeg_file)
            text_ids, text_mask = self._tokenize_text(cleaned_report)
            age_token, sex_token, race_token = self._load_demo_data(eeg_file)
            demo_token = torch.tensor([age_token, sex_token, race_token], dtype=torch.long)
            disease_token, disease_mask = self._load_disease_data(eeg_file)
            medication_token, medication_mask = self._load_medication_data(eeg_file)

        return {
            'eeg': eeg_t,
            'pos': self._positions_tensor.clone(),
            'batch_mask': batch_mask,
            'batch_unmask': batch_unmask,
            'text_ids': text_ids,
            'text_mask': text_mask,
            'demo_token': demo_token,
            'disease_token': disease_token,
            'medication_token': medication_token,
            'disease_mask': disease_mask,
            'medication_mask': medication_mask,
            'report_valid': report_valid,
            'eeg_file': eeg_file,
        }


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
    """Filter None items, collate the rest, attach n_skipped to the dict."""
    n_skipped = sum(1 for item in batch if item is None)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    try:
        collated = torch.utils.data.dataloader.default_collate(batch)
        collated['n_skipped'] = n_skipped
        return collated
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


def _mm_kwargs_from_args(args, mode):
    """Pull multimodal kwargs from args.data, falling back to defaults."""
    data_cfg = args.data
    return dict(
        disable_clip=getattr(data_cfg, 'disable_clip', False),
        report_dir=getattr(data_cfg, 'report_dir', DEFAULT_REPORT_DIR),
        tokenizer_model_name=getattr(data_cfg, 'tokenizer_model_name', "google/t5-v1_1-base"),
        tokenizer_cache_dir=getattr(data_cfg, 'tokenizer_cache_dir', DEFAULT_TOKENIZER_CACHE_DIR),
        max_text_length=getattr(data_cfg, 'max_text_length', 256),
        max_num_disease=getattr(data_cfg, 'max_num_disease', 30),
        max_num_medication=getattr(data_cfg, 'max_num_medication', 50),
        mode=mode,
    )


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
        **_mm_kwargs_from_args(args, mode='train' if shuffle else 'val'),
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
        fixed_start=not shuffle,
        **_mm_kwargs_from_args(args, mode='train' if shuffle else 'val'),
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
