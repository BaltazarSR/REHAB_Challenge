from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sequence_data import SequenceDataset

RANDOM_SEED = 14
N_CLASSES = 16
DEVICE = torch.device("cpu")  # CPU for reproducibility, same spirit as fixing random_state=14 elsewhere


def _masked_mean_pool(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """x: (batch, channels, time). Averages only over [0, length) per sample."""
    batch, channels, time_len = x.shape
    arange = torch.arange(time_len, device=x.device).unsqueeze(0)
    mask = (arange < lengths.unsqueeze(1)).unsqueeze(1).float()  # (batch, 1, time)
    summed = (x * mask).sum(dim=2)
    return summed / lengths.clamp(min=1).unsqueeze(1).float()


class _CNNBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return _masked_mean_pool(x, lengths)  # (batch, hidden_channels * 2)


class TwoBranchCNN(nn.Module):
    def __init__(self, hidden_channels: int = 32, n_classes: int = N_CLASSES):
        super().__init__()
        self.branch1 = _CNNBranch(6, hidden_channels)
        self.branch2 = _CNNBranch(6, hidden_channels)
        feat_dim = hidden_channels * 2 * 2  # two branches, each hidden_channels*2 wide
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_channels * 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, n_classes),
        )

    def forward(self, x1, len1, x2, len2):
        f1 = self.branch1(x1, len1)
        f2 = self.branch2(x2, len2)
        return self.head(torch.cat([f1, f2], dim=1))


class _LSTMBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (2 [directions], batch, hidden_size) -> concat forward/backward
        return torch.cat([h_n[0], h_n[1]], dim=1)


class TwoBranchLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, n_classes: int = N_CLASSES):
        super().__init__()
        self.branch1 = _LSTMBranch(6, hidden_size)
        self.branch2 = _LSTMBranch(6, hidden_size)
        feat_dim = hidden_size * 2 * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_size * 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, n_classes),
        )

    def forward(self, x1, len1, x2, len2):
        f1 = self.branch1(x1, len1)
        f2 = self.branch2(x2, len2)
        return self.head(torch.cat([f1, f2], dim=1))


class _TransformerBranch(nn.Module):
    def __init__(self, in_channels: int, max_len: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, time_len, _ = x.shape
        h = self.input_proj(x) + self.pos_embedding[:, :time_len, :]
        arange = torch.arange(time_len, device=x.device).unsqueeze(0)
        pad_mask = arange >= lengths.unsqueeze(1)  # True = padded position (masked out as a key)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        valid_mask = (~pad_mask).unsqueeze(-1).float()
        summed = (h * valid_mask).sum(dim=1)
        return summed / lengths.clamp(min=1).unsqueeze(1).float()


class TwoBranchTransformer(nn.Module):
    def __init__(self, max_len: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 64, n_classes: int = N_CLASSES):
        super().__init__()
        self.branch1 = _TransformerBranch(6, max_len, d_model, nhead, num_layers, dim_feedforward)
        self.branch2 = _TransformerBranch(6, max_len, d_model, nhead, num_layers, dim_feedforward)
        feat_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x1, len1, x2, len2):
        f1 = self.branch1(x1, len1)
        f2 = self.branch2(x2, len2)
        return self.head(torch.cat([f1, f2], dim=1))


def _class_weights(y: np.ndarray, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float32)


def _run_epoch(model, loader, optimizer, criterion, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    n = 0
    for x1, len1, x2, len2, y in loader:
        x1, len1, x2, len2, y = (t.to(DEVICE) for t in (x1, len1, x2, len2, y))
        if train:
            optimizer.zero_grad()
        logits = model(x1, len1, x2, len2)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


@torch.no_grad()
def _predict(model, loader) -> np.ndarray:
    model.eval()
    preds = []
    for x1, len1, x2, len2, y in loader:
        x1, len1, x2, len2 = (t.to(DEVICE) for t in (x1, len1, x2, len2))
        logits = model(x1, len1, x2, len2)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _fit(model_cls, model_kwargs: dict, train_ds: SequenceDataset, val_ds: SequenceDataset,
         y_train: np.ndarray, lr: float, max_epochs: int, patience: int, batch_size: int = 64):
    torch.manual_seed(RANDOM_SEED)
    model = model_cls(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train).to(DEVICE))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    y_val = val_ds.y.numpy()

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        _run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_preds = _predict(model, val_loader)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_f1


def _refit_full(model_cls, model_kwargs: dict, train_ds: SequenceDataset, y_train: np.ndarray,
                 lr: float, n_epochs: int, batch_size: int = 64):
    torch.manual_seed(RANDOM_SEED)
    model = model_cls(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train).to(DEVICE))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    for _ in range(n_epochs):
        _run_epoch(model, train_loader, optimizer, criterion, train=True)
    return model


def train_and_evaluate(model_cls, param_grid: list[dict], X1_train, len1_train, X2_train, len2_train, y_train,
                        X1_test, len1_test, X2_test, len2_test, y_test,
                        max_epochs: int = 30, patience: int = 5, batch_size: int = 64) -> dict:
    """Small param grid, each config selected via a stratified 85/15 val split of the
    training data (early-stopped on val macro-F1); best config is refit on the FULL
    training split for `best_epoch` epochs, mirroring GridSearchCV(refit=True), then
    evaluated once on the untouched test split."""
    idx = np.arange(len(y_train))
    sub_idx, val_idx = train_test_split(idx, test_size=0.15, stratify=y_train, random_state=RANDOM_SEED)

    def subset(X1, l1, X2, l2, y, ix):
        return SequenceDataset(X1[ix], l1[ix], X2[ix], l2[ix], y[ix])

    sub_ds = subset(X1_train, len1_train, X2_train, len2_train, y_train, sub_idx)
    val_ds = subset(X1_train, len1_train, X2_train, len2_train, y_train, val_idx)
    y_sub = y_train[sub_idx]

    t0 = time.time()
    best_overall = {"val_f1": -1.0}
    for params in param_grid:
        lr = params.pop("lr")
        _, best_epoch, val_f1 = _fit(model_cls, params, sub_ds, val_ds, y_sub, lr, max_epochs, patience, batch_size)
        print(f"  {model_cls.__name__} params={params} lr={lr} -> val f1_macro={val_f1:.3f} @ epoch {best_epoch}")
        if val_f1 > best_overall["val_f1"]:
            best_overall = {"val_f1": val_f1, "params": dict(params), "lr": lr, "best_epoch": best_epoch}
        params["lr"] = lr  # restore for repeated calls with the same grid object

    search_time = time.time() - t0

    full_train_ds = SequenceDataset(X1_train, len1_train, X2_train, len2_train, y_train)
    model = _refit_full(model_cls, best_overall["params"], full_train_ds, y_train,
                         best_overall["lr"], best_overall["best_epoch"], batch_size)

    test_ds = SequenceDataset(X1_test, len1_test, X2_test, len2_test, y_test)
    t0 = time.time()
    test_preds = _predict(model, DataLoader(test_ds, batch_size=batch_size, shuffle=False))
    predict_time = time.time() - t0

    return {
        "model": model,
        "best_params": best_overall["params"],
        "best_lr": best_overall["lr"],
        "val_f1_macro": best_overall["val_f1"],
        "search_time_s": search_time,
        "predict_time_s": predict_time,
        "test_preds": test_preds,
    }
