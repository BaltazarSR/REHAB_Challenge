"""
Funciones para leer una muestra puntual de `d04_transformed_data/` (la data
final, lista para analisis/modelado) usando manifest.csv para saber cual es
su region valida (`valid_length`).

A diferencia de inspect_utils.py (que compara raw/clean/transformed para
verificar que la limpieza y la normalizacion funcionaron), este modulo
asume que ya confiamos en el pipeline y solo trabaja con la data final.

Contexto de los datos: ver Data/README.md
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import etl_utils as u

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "Data", "Rehab_exercise"))
TRANSFORMED_SUBDIR = "d04_transformed_data"


def load_manifest(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "manifest.csv"))


def load_movement(movement_id: int, sensor_id: int, data_dir: str = DEFAULT_DATA_DIR) -> np.ndarray:
    """Carga <movementID>_<sensorID>.npy de d04_transformed_data."""
    fname = f"{movement_id:03d}_{sensor_id}.npy"
    return np.load(os.path.join(data_dir, TRANSFORMED_SUBDIR, fname))


def get_manifest_row(manifest_df: pd.DataFrame, movement_id: int, sample_index: int) -> pd.Series:
    row = manifest_df[
        (manifest_df["movement_id"] == movement_id)
        & (manifest_df["sample_index"] == sample_index)
    ]
    if row.empty:
        raise ValueError(
            f"No hay fila en el manifest para movement_id={movement_id}, sample_index={sample_index}."
        )
    return row.iloc[0]


def resolve_channel(sensor_id: int, channel: int | str) -> tuple[int, str]:
    """Acepta indice de canal (int) o nombre (str, ej. 'f1') y regresa ambos."""
    names = u.channels_for(sensor_id)
    if isinstance(channel, str):
        if channel not in names:
            raise ValueError(f"canal {channel!r} no existe en sensor {sensor_id} (opciones: {names})")
        return names.index(channel), channel
    return channel, names[channel]


def get_sample_channel(manifest_df: pd.DataFrame, movement_id: int, sample_index: int, sensor_id: int,
                        channel: int | str, data_dir: str = DEFAULT_DATA_DIR, trim_to_valid: bool = True):
    """Regresa (senial_1d, valid_len, nombre_canal) para una muestra/canal de d04_transformed_data."""
    row = get_manifest_row(manifest_df, movement_id, sample_index)
    vlen = int(row[f"valid_length_sensor{sensor_id}"])
    ch_idx, ch_name = resolve_channel(sensor_id, channel)
    arr = load_movement(movement_id, sensor_id, data_dir)
    sig = arr[sample_index, :, ch_idx]
    if trim_to_valid:
        sig = sig[:vlen]
    return sig, vlen, ch_name


def plot_sample_channel(manifest_df: pd.DataFrame, movement_id: int, sample_index: int, sensor_id: int,
                         channel: int | str, data_dir: str = DEFAULT_DATA_DIR, ax=None):
    """Grafica una muestra/canal de d04_transformed_data."""
    sig, vlen, ch_name = get_sample_channel(manifest_df, movement_id, sample_index, sensor_id, channel, data_dir)
    movement_name = u.MOVEMENT_LIBRARY[movement_id][0]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    ax.plot(sig)
    ax.set_title(f"mov {movement_id} ({movement_name}), muestra {sample_index}, {ch_name}")
    ax.set_xlabel("timestep")
    ax.set_ylabel("z-score")
    return ax


def plot_sample_all_channels(manifest_df: pd.DataFrame, movement_id: int, sample_index: int, sensor_id: int,
                              data_dir: str = DEFAULT_DATA_DIR):
    """Grid 2x3 con los 6 canales de un sensor para una muestra de d04_transformed_data."""
    channels = u.channels_for(sensor_id)
    movement_name = u.MOVEMENT_LIBRARY[movement_id][0]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ch_idx, axi in zip(range(len(channels)), axes.ravel()):
        sig, vlen, ch_name = get_sample_channel(manifest_df, movement_id, sample_index, sensor_id, ch_idx, data_dir)
        axi.plot(sig)
        axi.set_title(ch_name)
        axi.set_xlabel("timestep")

    fig.suptitle(f"mov {movement_id} ({movement_name}), sensor {sensor_id}, muestra {sample_index}")
    fig.tight_layout()
    return axes
