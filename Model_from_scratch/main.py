from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import etl_utils as u
from utils import manifest_utils as mu

N_SEGMENTS = 10
MIN_POINTS_PER_SEGMENT = 3
MIN_VALID_LENGTH = N_SEGMENTS * MIN_POINTS_PER_SEGMENT 

STATS = ("median", "mean", "std", "min", "max", "skew")


def segment_bounds(valid_length: int) -> list[tuple[int, int]]:
    edges = [round(i * valid_length / N_SEGMENTS) for i in range(N_SEGMENTS + 1)]
    return list(zip(edges[:-1], edges[1:]))


def segment_stats(values: np.ndarray) -> dict[str, float]:
    mean = float(values.mean())
    std = float(values.std())
    skew = float(np.mean(((values - mean) / std) ** 3)) if std > 0 else 0.0
    return {
        "median": float(np.median(values)),
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "skew": skew,
    }


def feature_columns() -> list[str]:
    return [
        f"seg{i:02d}_{ch}_{stat}"
        for i in range(N_SEGMENTS)
        for sensor_id in (1, 2)
        for ch in u.channels_for(sensor_id)
        for stat in STATS
    ]


def extract_sample_features(arr1: np.ndarray, arr2: np.ndarray, sample_index: int, vlen1: int, vlen2: int) -> dict[str, float]:
    feats: dict[str, float] = {}
    for sensor_id, arr, vlen in ((1, arr1, vlen1), (2, arr2, vlen2)):
        channels = u.channels_for(sensor_id)
        for seg_i, (start, end) in enumerate(segment_bounds(vlen)):
            for ch_idx, ch_name in enumerate(channels):
                stats = segment_stats(arr[sample_index, start:end, ch_idx])
                for stat_name, val in stats.items():
                    feats[f"seg{seg_i:02d}_{ch_name}_{stat_name}"] = val
    return feats


def build_feature_table() -> pd.DataFrame:
    manifest = mu.load_manifest()

    too_short = (manifest["valid_length_sensor1"] < MIN_VALID_LENGTH) | \
                (manifest["valid_length_sensor2"] < MIN_VALID_LENGTH)
    dropped = manifest[too_short]
    if len(dropped):
        print(f"Dropping {len(dropped)} sample(s) with valid_length < {MIN_VALID_LENGTH}:")
        for _, r in dropped.iterrows():
            print(f"  movement_id={r.movement_id}, sample_index={r.sample_index}, "
                  f"valid_length_sensor1={r.valid_length_sensor1}, "
                  f"valid_length_sensor2={r.valid_length_sensor2}")
    manifest = manifest[~too_short]

    columns = feature_columns()
    rows: list[dict[str, float]] = []
    movement_ids: list[int] = []
    sample_indices: list[int] = []

    for movement_id, group in manifest.groupby("movement_id"):
        arr1 = mu.load_movement(int(movement_id), 1)
        arr2 = mu.load_movement(int(movement_id), 2)
        for r in group.itertuples():
            feats = extract_sample_features(
                arr1, arr2, r.sample_index,
                int(r.valid_length_sensor1), int(r.valid_length_sensor2),
            )
            rows.append(feats)
            movement_ids.append(int(movement_id))
            sample_indices.append(int(r.sample_index))

    df = pd.DataFrame(rows, columns=columns)
    df.insert(0, "sample_index", sample_indices)
    df.insert(0, "movement_id", movement_ids)
    return df


# ---------------------------------------------------------------------------
# softmax regression
# ---------------------------------------------------------------------------

RANDOM_SEED = 14
TEST_FRACTION = 0.2
LEARNING_RATE = 0.05
N_ITERATIONS = 2000


def train_test_split(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    n_test = round(n_samples * TEST_FRACTION)
    return idx[n_test:], idx[:n_test]


def standardize(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    return (X - mean) / std, mean, std


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)


class SoftmaxRegression:

    def __init__(self):
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "SoftmaxRegression":
        n_samples, n_features = X.shape
        Y = one_hot(y, n_classes)
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        for _ in range(N_ITERATIONS):
            probs = softmax(X @ self.W + self.b)
            error = probs - Y

            grad_W = (X.T @ error) / n_samples
            grad_b = error.mean(axis=0)

            self.W -= LEARNING_RATE * grad_W
            self.b -= LEARNING_RATE * grad_b

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = softmax(X @ self.W + self.b)
        return probs.argmax(axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def train_and_evaluate(df: pd.DataFrame) -> SoftmaxRegression:
    feature_cols = [c for c in df.columns if c not in ("movement_id", "sample_index")]
    X_all = df[feature_cols].to_numpy(dtype=np.float64)

    classes = np.sort(df["movement_id"].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_all = df["movement_id"].map(class_to_idx).to_numpy()

    train_idx, test_idx = train_test_split(len(y_all))
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    X_train, mean, std = standardize(X_train)
    X_test, _, _ = standardize(X_test, mean, std)

    model = SoftmaxRegression().fit(X_train, y_train, n_classes=len(classes))

    train_acc = accuracy(y_train, model.predict(X_train))
    test_acc = accuracy(y_test, model.predict(X_test))
    print(f"train accuracy: {train_acc:.3f}  ({len(y_train)} samples)")
    print(f"test accuracy:  {test_acc:.3f}  ({len(y_test)} samples)")

    return model


def main() -> None:
    features_path = os.path.join(_THIS_DIR, "features.csv")
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
    else:
        df = build_feature_table()
        df.to_csv(features_path, index=False)
        print(f"Wrote {df.shape[0]} rows x {df.shape[1]} cols to {features_path}")

    train_and_evaluate(df)


if __name__ == "__main__":
    main()
