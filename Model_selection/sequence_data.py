from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import manifest_utils as mu

# Loads the raw (880, 6) x 2 sensor sequences straight from manifest.csv /
# d04_transformed_data/, independent of Model_from_scratch. Sensor1 and sensor2
# are kept as two separate arrays (not merged into one 12-channel tensor) because
# their valid_length values diverge independently per sample -- a shared time
# mask over a merged tensor would be wrong (see Data/README.md).


def load_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest = mu.load_manifest()

    n = len(manifest)
    X1 = np.zeros((n, 880, 6), dtype=np.float32)
    X2 = np.zeros((n, 880, 6), dtype=np.float32)
    len1 = np.zeros(n, dtype=np.int64)
    len2 = np.zeros(n, dtype=np.int64)
    movement_id = np.zeros(n, dtype=np.int64)
    sample_index = np.zeros(n, dtype=np.int64)

    row = 0
    for mid, group in manifest.groupby("movement_id"):
        arr1 = mu.load_movement(int(mid), 1)
        arr2 = mu.load_movement(int(mid), 2)
        for r in group.itertuples():
            X1[row] = arr1[r.sample_index]
            X2[row] = arr2[r.sample_index]
            len1[row] = int(r.valid_length_sensor1)
            len2[row] = int(r.valid_length_sensor2)
            movement_id[row] = int(mid)
            sample_index[row] = int(r.sample_index)
            row += 1

    y = movement_id.copy()
    return X1, len1, X2, len2, y, movement_id, sample_index


def downsample(X: np.ndarray, lengths: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Strided subsampling in time (no averaging/stat-engineering, just fewer of the
    same raw readings) -- makes CPU training of the recurrent branch tractable.
    880 timesteps at 50Hz is far finer than rehab movements need; stride=8 keeps
    ~6 samples/sec, still well above the movement's dynamics."""
    X_ds = np.ascontiguousarray(X[:, ::stride, :])
    new_len = X_ds.shape[1]
    lengths_ds = np.clip(np.ceil(lengths / stride).astype(np.int64), 1, new_len)
    return X_ds, lengths_ds


class SequenceDataset(Dataset):
    def __init__(self, X1: np.ndarray, len1: np.ndarray, X2: np.ndarray, len2: np.ndarray, y: np.ndarray):
        self.X1 = torch.from_numpy(X1)
        self.len1 = torch.from_numpy(len1)
        self.X2 = torch.from_numpy(X2)
        self.len2 = torch.from_numpy(len2)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.X1[idx], self.len1[idx], self.X2[idx], self.len2[idx], self.y[idx]
