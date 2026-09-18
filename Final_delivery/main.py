"""
Streamlit UI for the REHAB movement classifier -- the "Solution" deliverable.

Two sections, picked from the sidebar:
- "Try the Model": loads the trained Transformer (transformer_model.pt) and
  runs it on a real sample from the dataset, showing the predicted movement
  vs. the recorded one.
- "About the Project": informational placeholder, content to be filled in
  later.

Run with: streamlit run Final_delivery/main.py
"""

from __future__ import annotations

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import etl_utils as u
from utils import manifest_utils as mu
from Transformer import TwoBranchTransformer

sns.set_theme(style="whitegrid", palette="deep")

MODEL_PATH = os.path.join(_THIS_DIR, "transformer_model.pt")


# ---------------------------------------------------------------------------
# Cached resources: the trained model and the manifest
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = TwoBranchTransformer(
        max_len=ckpt["max_len"], n_classes=ckpt["n_classes"], **ckpt["model_kwargs"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data
def load_manifest_df() -> pd.DataFrame:
    return mu.load_manifest()


# ---------------------------------------------------------------------------
# Inference: same preprocessing as training (stride downsample), on one sample
# ---------------------------------------------------------------------------


def downsample(x: np.ndarray, valid_len: int, stride: int) -> tuple[np.ndarray, int]:
    x_ds = np.ascontiguousarray(x[::stride, :])
    new_len = x_ds.shape[0]
    len_ds = int(np.clip(np.ceil(valid_len / stride), 1, new_len))
    return x_ds, len_ds


def plot_example_repetition(manifest_df: pd.DataFrame, movement_id: int):
    """Plots both sensors' raw channels for one example repetition of `movement_id`,
    shading the padded (non-real) region so a first-time reader can see what the
    input signal actually looks like, including the padding concept explained above."""
    row = manifest_df[manifest_df["movement_id"] == movement_id].iloc[0]
    sample_index = int(row["sample_index"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sensor_id, title in ((axes[0], 1, "Limb sensor"), (axes[1], 2, "Glove sensor")):
        arr = mu.load_movement(movement_id, sensor_id)[sample_index]
        valid_len = int(row[f"valid_length_sensor{sensor_id}"])
        channels = u.channels_for(sensor_id)

        df_long = pd.DataFrame(arr, columns=channels)
        df_long["timestep"] = np.arange(len(df_long))
        df_long = df_long.melt(id_vars="timestep", var_name="channel", value_name="value")

        sns.lineplot(data=df_long, x="timestep", y="value", hue="channel", linewidth=1, ax=ax)
        ax.axvspan(valid_len, len(arr), color="lightgray", alpha=0.5, label="padding")
        ax.set_title(title)
        ax.set_xlabel("timestep (1/50 sec each)")
        ax.set_ylabel("z-scored reading (unitless)")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Example repetition -- movement {movement_id} ({row['movement_name']})")
    fig.tight_layout()
    return fig


def predict_sample(model, ckpt, manifest_df: pd.DataFrame, movement_id: int, sample_index: int):
    row = mu.get_manifest_row(manifest_df, movement_id, sample_index)
    stride = ckpt["stride"]

    arr1 = mu.load_movement(movement_id, 1)[sample_index]
    arr2 = mu.load_movement(movement_id, 2)[sample_index]
    x1_ds, len1_ds = downsample(arr1, int(row["valid_length_sensor1"]), stride)
    x2_ds, len2_ds = downsample(arr2, int(row["valid_length_sensor2"]), stride)

    x1 = torch.from_numpy(x1_ds).float().unsqueeze(0)
    x2 = torch.from_numpy(x2_ds).float().unsqueeze(0)
    len1 = torch.tensor([len1_ds])
    len2 = torch.tensor([len2_ds])

    with torch.no_grad():
        logits = model(x1, len1, x2, len2)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    predicted_id = int(np.argmax(probs))
    return predicted_id, probs, row


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


st.set_page_config(page_title="REHAB Movement Classifier", layout="centered")

page = st.sidebar.radio("Section", ["Try the Model", "About the Project"])

if page == "Try the Model":
    st.title("Try the Model")
    st.write("Pick a recorded repetition from the dataset and see what the trained Transformer predicts for it.")

    model, ckpt = load_model()
    manifest_df = load_manifest_df()

    movements = manifest_df[["movement_id", "movement_name"]].drop_duplicates().sort_values("movement_id")
    movement_options = {f"{int(r.movement_id)} -- {r.movement_name}": int(r.movement_id) for r in movements.itertuples()}

    if st.button("Random sample"):
        row = manifest_df.sample(1).iloc[0]
        st.session_state["movement_choice"] = f"{int(row.movement_id)} -- {row.movement_name}"
        st.session_state["sample_index"] = int(row.sample_index)

    movement_choice = st.selectbox(
        "Movement", list(movement_options.keys()), key="movement_choice"
    )
    movement_id = movement_options[movement_choice]

    sample_indices = sorted(manifest_df[manifest_df["movement_id"] == movement_id]["sample_index"].tolist())
    if "sample_index" not in st.session_state or st.session_state["sample_index"] not in sample_indices:
        st.session_state["sample_index"] = sample_indices[0]
    sample_index = st.selectbox("Sample index", sample_indices, key="sample_index")

    if st.button("Classify", type="primary"):
        predicted_id, probs, row = predict_sample(model, ckpt, manifest_df, movement_id, sample_index)
        predicted_name = movements[movements["movement_id"] == predicted_id]["movement_name"].iloc[0]
        true_name = row["movement_name"]

        correct = predicted_id == movement_id
        st.metric(
            label="Prediction" + (" (correct)" if correct else " (incorrect)"),
            value=f"{predicted_id} -- {predicted_name}",
            delta=f"confidence {probs[predicted_id]:.1%}",
        )
        st.write(f"**Recorded label:** {movement_id} -- {true_name}")

        prob_df = pd.DataFrame({
            "movement_id": range(len(probs)),
            "probability": probs,
        }).set_index("movement_id")
        st.bar_chart(prob_df)

    st.caption(
        "Samples are drawn from the full dataset, not exclusively the model's held-out "
        "test split, so a correct prediction here isn't the same as the reported test metrics."
    )

else:
    _, ckpt = load_model()
    manifest_df = load_manifest_df()

    st.title("About the Project")

    st.header("Problem Definition")
    st.markdown(
        "Stroke survivors are often prescribed rehabilitation exercises to do at home, but "
        "without a therapist present, there's no way to verify the prescribed movement was "
        "actually performed correctly -- or at all. This project builds a classifier that looks "
        "at a single repetition's raw wearable-sensor recording and identifies which of 16 "
        "rehabilitation exercises it was, as a building block toward automated, at-home exercise "
        "verification."
    )

    st.header("Approach to solve it")

    st.subheader("1. Understanding the data")
    st.markdown(
        "- **Source:** Wearable sensor kinematic "
        "recordings from 120 post-stroke patients performing 16 standardized rehab exercises "
        "(12 upper-limb, 4 lower-limb).\n"
        "- **Input:** two sensors per repetition, a limb IMU (6 channels: pitch/yaw/roll x2) "
        "and a glove (6 channels: 5 finger-flexion sensors + wrist pitch) each up to 880 "
        "timesteps, but only a variable-length prefix is real signal (the rest is padding).\n"
        "- **Output:** one of 16 movement classes (e.g. \"Ball gripping,\" \"Wrist flexion and "
        "extension,\" \"Knee flexion and extension\").\n"
        "- **What makes it hard:** class sizes are imbalanced (~170-380 repetitions/class), the "
        "two sensors' valid signal lengths vary independently of each other for the same "
        "repetition.\n"
        "- After cleaning: **3,884 repetitions** used for modeling."
    )

    st.markdown("**All 16 movements:**")
    movements_table = (
        manifest_df[["movement_id", "movement_name", "limb"]]
        .drop_duplicates()
        .sort_values("movement_id")
        .set_index("movement_id")
    )
    st.table(movements_table)

    st.markdown("**What one repetition looks like:**")
    st.pyplot(plot_example_repetition(manifest_df, movement_id=7))
    st.caption(
        "Each colored line is one sensor channel, sampled 50 times per second. The shaded "
        "gray area is padding: the sensor's real recording ends there (its `valid_length`), "
        "and everything after is filler with no movement information, not part of the signal."
    )

    st.subheader("2. Data cleaning (ETL)")
    st.markdown(
        "- **Empty repetitions:** 142 of 9,232 raw samples were 100% zero-padding (no real "
        "signal), dropped entirely.\n"
        "- **Exact duplicates:** 1,438 samples were byte-for-byte identical to another sample, "
        "this is statistically impossible for independently-recorded analog sensor data, so treated "
        "as an export-pipeline artifact and deduplicated.\n"
        "- **Physically impossible angle values:** the yaw/roll channels are computed via "
        "`arctan2`, whose exact mathematical range is `(-180 deg, 180 deg]`, yet raw values "
        "reached up to +/-1,178 deg. 47,806 individual points (limb sensor) + 674 (glove "
        "wrist-pitch) were repaired by linear interpolation within the valid region.\n"
        "- **Order matters:** angle repair happens *before* normalization, so one corrupted "
        "~887 deg reading can't distort the mean/std used to normalize an otherwise-valid "
        "sample.\n"
        "- **Smoothing:** a 10-sample moving average, computed only within each sample's valid "
        "region, so the zero-padded tail doesn't bias the last real points.\n"
        "- **Normalization:** zero-mean/unit-variance z-scoring, per sample and per channel, "
        "computed only over the valid region. The dataset's *original* "
        "normalization (computed over the full 880-timestep window, padding included) was found "
        "to leak a nonzero constant into the padding once a sample wasn't perfectly zero-mean.\n"
        "- Output: cleaned + smoothed + normalized arrays, plus a `manifest.csv` recording each "
        "sample's real signal length, so padding is always excludable by lookup rather than by "
        "guessing from the values."
    )

    st.subheader("3. Model selection")
    st.markdown(
        "- Each repetition is a variable-length, multi-channel time series, not a fixed-size "
        "feature vector so this rules out classic tabular models (logistic regression, SVM, "
        "random forest) without heavy feature engineering first.\n"
        "- Three sequence-native architectures were compared, each with a \"two-branch\" design "
        "(limb and glove sensors processed independently, since their valid lengths diverge "
        "independently per repetition): a two-branch **1D-CNN** (local temporal patterns), a "
        "two-branch **LSTM** (long-range dependencies via recurrence), and a two-branch "
        "**Transformer encoder** (self-attention across the whole sequence at once).\n"
        "- The Transformer came out ahead on a held-out comparison (test macro-F1 0.915), "
        "clearly ahead of the LSTM and modestly ahead of the CNN, and was carried forward as the "
        "final architecture."
    )

    st.subheader("4. Final architecture")
    st.markdown(
        "- Two-branch Transformer encoder. Each sensor's 6 raw channels are projected into a "
        "richer per-timestep representation, combined with a learned positional embedding.\n"
        "- Self-attention layers let every timestep weigh every other valid timestep directly; "
        "padded timesteps are explicitly masked out of attention, so they're never treated as "
        "real signal.\n"
        "- Each branch's output is pooled over only its valid timesteps, the two branches' "
        "outputs are concatenated, and a small classification head produces the final 16-way "
        "prediction."
    )

    st.subheader("5. Training process")
    st.markdown(
        "- A single stratified 70% train / 15% validation / 15% test split.\n"
        "- Regularization: dropout, L2 weight decay, early stopping on validation macro-F1, and "
        "class-weighted loss (so the imbalance between classes doesn't get ignored).\n"
        "- Hyperparameter search: a 4-config grid varying model capacity and regularization "
        "strength together, the highest-capacity config with the strongest "
        "regularization won."
    )

    st.header("Results, supported by metrics")

    st.markdown("**Hyperparameter search (validation macro-F1):**")
    search_df = pd.DataFrame([
        {"Capacity": "d_model=64, heads=8, layers=3", "Dropout": 0.4, "Weight decay": 1e-4, "Val macro-F1": 0.887},
        {"Capacity": "d_model=64, heads=4, layers=2", "Dropout": 0.3, "Weight decay": 0.0, "Val macro-F1": 0.854},
        {"Capacity": "d_model=64, heads=4, layers=2", "Dropout": 0.5, "Weight decay": 1e-4, "Val macro-F1": 0.813},
        {"Capacity": "d_model=32, heads=4, layers=2", "Dropout": 0.3, "Weight decay": 1e-4, "Val macro-F1": 0.806},
    ])
    st.table(search_df.set_index("Capacity"))

    st.markdown("**Held-out test set (583 repetitions, never touched during training or tuning):**")
    metrics_df = pd.DataFrame([
        {"Metric": "Accuracy", "Score": ckpt["test_accuracy"]},
        {"Metric": "Macro-F1", "Score": ckpt["test_f1_macro"]},
    ])
    st.table(metrics_df.set_index("Metric"))

    st.markdown(
        "Per-class F1 mostly 0.83-1.00; the weakest class is \"Shoulder touch training\" "
        "(F1 0.66) -- the model over-predicts this class rather than missing it, as shown below."
    )

    confusion_matrix_path = os.path.join(_THIS_DIR, "figures", "confusion_matrix.png")
    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption="Test confusion matrix (row-normalized)")
