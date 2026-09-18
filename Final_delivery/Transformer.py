"""
Final model definition for the final-delivery pipeline.

Defines the two-branch Transformer encoder that Model_selection's comparison
(CNN vs LSTM vs Transformer) picked as the best architecture (test macro-F1
0.915, see Model_selection/model_selection.ipynb section 5).

Data loading and the configuration/training/validation/regularization/
hyperparameter search for this architecture live in `model_training.ipynb`
in this folder, where the process is visible end-to-end.
TwoBranchTransformer's defaults below are the winning hyperparameters that
notebook selected -- this file is the final model, ready to be instantiated
with no arguments beyond `max_len`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 16

# ---------------------------------------------------------------------------
# Model: two-branch Transformer encoder
# ---------------------------------------------------------------------------


class _TransformerBranch(nn.Module):
    def __init__(self, in_channels: int, max_len: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int, dropout: float):
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
    def __init__(self, max_len: int, d_model: int = 64, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 128, dropout: float = 0.4, n_classes: int = N_CLASSES):
        super().__init__()
        self.branch1 = _TransformerBranch(6, max_len, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.branch2 = _TransformerBranch(6, max_len, d_model, nhead, num_layers, dim_feedforward, dropout)
        feat_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x1, len1, x2, len2):
        f1 = self.branch1(x1, len1)
        f2 = self.branch2(x2, len2)
        return self.head(torch.cat([f1, f2], dim=1))
