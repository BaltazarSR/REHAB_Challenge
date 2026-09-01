"""
Utilidades para inspeccionar/graficar los datos YA generados por
ETL_Cleaning.ipynb (d03_clean_data, d04_transformed_data, manifest.csv),
sin volver a correr el pipeline de limpieza/transformacion.

Se apoyan en manifest.csv como fuente de verdad del valid_length de cada
muestra (ver Data/README.md: no se debe re-detectar el padding buscando
ceros, porque los datos normalizados pueden cruzar por cero legitimamente).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import etl_utils as u

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "Data", "Rehab_exercise"))

STAGE_DIRS = {
    "raw": "d01_raw_data",                  # crudo, tal como lo entrega el dataset
    "authors_processed": "d02_processed_data",  # filtrado/normalizado por los autores originales
    "clean": "d03_clean_data",              # nuestro: sin vacios/duplicados, angulos reparados
    "transformed": "d04_transformed_data",  # nuestro: clean + suavizado + normalizado
}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_manifest(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "manifest.csv"))


def load_stage_array(stage: str, movement_id: int, sensor_id: int,
                      data_dir: str = DEFAULT_DATA_DIR) -> np.ndarray:
    """Carga <movementID>_<sensorID>.npy de la etapa pedida (ver STAGE_DIRS)."""
    if stage not in STAGE_DIRS:
        raise ValueError(f"stage debe ser uno de {list(STAGE_DIRS)}, recibido: {stage!r}")
    fname = f"{movement_id:03d}_{sensor_id}.npy"
    return np.load(os.path.join(data_dir, STAGE_DIRS[stage], fname))


def get_manifest_row(manifest_df: pd.DataFrame, movement_id: int, sample_index: int) -> pd.Series:
    row = manifest_df[
        (manifest_df["movement_id"] == movement_id)
        & (manifest_df["sample_index"] == sample_index)
    ]
    if row.empty:
        raise ValueError(
            f"No hay fila en el manifest para movement_id={movement_id}, sample_index={sample_index} "
            "(la muestra pudo haber sido eliminada en la limpieza, o el indice no existe)."
        )
    return row.iloc[0]


def sample_index_from_raw_index(manifest_df: pd.DataFrame, movement_id: int, raw_index: int) -> int:
    """Traduce una fila de d01_raw_data/d02_processed_data (ej. la que usan las
    figuras de EDA_Graphs.ipynb, que indexan el crudo directo) al sample_index
    correspondiente en d03_clean_data/d04_transformed_data, para poder pasarlo
    al resto de las funciones de este modulo (get_sample_channel, check_normalization,
    plot_sample_channel, plot_sample_all_channels, get_manifest_row)."""
    row = manifest_df[
        (manifest_df["movement_id"] == movement_id)
        & (manifest_df["orig_index_in_raw"] == raw_index)
    ]
    if row.empty:
        raise ValueError(
            f"No hay fila en el manifest con movement_id={movement_id}, orig_index_in_raw={raw_index} "
            "(esa muestra cruda fue eliminada en la limpieza: vacia o duplicada)."
        )
    return int(row.iloc[0]["sample_index"])


def valid_length(manifest_row: pd.Series, sensor_id: int) -> int:
    return int(manifest_row[f"valid_length_sensor{sensor_id}"])


# Etapas cuyo array NO fue reindexado por la limpieza (misma cantidad de filas
# y mismo orden que el dataset original): ahi hay que indexar con
# orig_index_in_raw, no con sample_index. d03/d04 SI fueron reindexados (se
# eliminaron filas vacias/duplicadas), asi que sample_index los indexa
# directamente -- ver ETL_Cleaning.ipynb celda 26 (manifest_df construido con
# sample_index = posicion dentro del array ya limpio).
RAW_ALIGNED_STAGES = {"raw", "authors_processed"}


def row_index_for_stage(stage: str, manifest_row: pd.Series, sample_index: int) -> int:
    if stage in RAW_ALIGNED_STAGES:
        return int(manifest_row["orig_index_in_raw"])
    return sample_index


def resolve_channel(sensor_id: int, channel: int | str) -> tuple[int, str]:
    """Acepta indice de canal (int) o nombre (str, ej. 'f1') y regresa ambos."""
    names = u.channels_for(sensor_id)
    if isinstance(channel, str):
        if channel not in names:
            raise ValueError(f"canal {channel!r} no existe en sensor {sensor_id} (opciones: {names})")
        return names.index(channel), channel
    return channel, names[channel]


# ---------------------------------------------------------------------------
# Consulta de una muestra/canal puntual
# ---------------------------------------------------------------------------

def get_sample_channel(stage: str, movement_id: int, sample_index: int, sensor_id: int,
                        channel: int | str, manifest_df: pd.DataFrame | None = None,
                        data_dir: str = DEFAULT_DATA_DIR, trim_to_valid: bool = True):
    """Regresa (senial_1d, valid_len, nombre_canal) para una muestra/canal de una etapa guardada.

    `sample_index` es siempre el indice en el espacio "limpio" (el de
    manifest.csv / d03 / d04). Para `stage="raw"` o `"authors_processed"` se
    traduce automaticamente a `orig_index_in_raw`, porque esas etapas no
    fueron reindexadas al eliminar muestras vacias/duplicadas -- indexar
    directo con `sample_index` ahi apuntaria a una muestra fisica distinta.
    """
    if manifest_df is None:
        manifest_df = load_manifest(data_dir)
    row = get_manifest_row(manifest_df, movement_id, sample_index)
    vlen = valid_length(row, sensor_id)
    ch_idx, ch_name = resolve_channel(sensor_id, channel)
    arr = load_stage_array(stage, movement_id, sensor_id, data_dir)
    array_row = row_index_for_stage(stage, row, sample_index)
    sig = arr[array_row, :, ch_idx]
    if trim_to_valid:
        sig = sig[:vlen]
    return sig, vlen, ch_name


# ---------------------------------------------------------------------------
# Verificacion numerica de la normalizacion
# ---------------------------------------------------------------------------

def check_normalization(movement_id: int, sample_index: int, sensor_id: int, channel: int | str,
                         manifest_df: pd.DataFrame | None = None, data_dir: str = DEFAULT_DATA_DIR,
                         mean_tol: float = 1e-3, std_tol: float = 1e-3) -> dict:
    """Compara clean vs. transformed para una muestra/canal puntual y evalua si
    la normalizacion z-score (media 0, desviacion 1 sobre la region valida) se
    aplico correctamente, y si el padding se mantuvo en 0 fuera de esa region."""
    if manifest_df is None:
        manifest_df = load_manifest(data_dir)

    clean_sig, vlen, ch_name = get_sample_channel(
        "clean", movement_id, sample_index, sensor_id, channel, manifest_df, data_dir
    )
    trans_sig, _, _ = get_sample_channel(
        "transformed", movement_id, sample_index, sensor_id, channel, manifest_df, data_dir
    )
    trans_full = load_stage_array("transformed", movement_id, sensor_id, data_dir)[
        sample_index, :, resolve_channel(sensor_id, channel)[0]
    ]
    padding_is_zero = bool(np.allclose(trans_full[vlen:], 0.0))

    trans_mean, trans_std = float(trans_sig.mean()), float(trans_sig.std())
    result = {
        "movement_id": movement_id,
        "movement_name": u.MOVEMENT_LIBRARY[movement_id][0],
        "sample_index": sample_index,
        "sensor_id": sensor_id,
        "channel": ch_name,
        "valid_length": vlen,
        "clean_mean": float(clean_sig.mean()),
        "clean_std": float(clean_sig.std()),
        "transformed_mean": trans_mean,
        "transformed_std": trans_std,
        "mean_ok": abs(trans_mean) < mean_tol,
        "std_ok": abs(trans_std - 1.0) < std_tol,
        "padding_is_zero": padding_is_zero,
    }
    result["ok"] = result["mean_ok"] and result["std_ok"] and result["padding_is_zero"]
    return result


def print_normalization_check(result: dict) -> None:
    status = "OK" if result["ok"] else "REVISAR"
    print(
        f"[{status}] movimiento {result['movement_id']} ({result['movement_name']}), "
        f"muestra {result['sample_index']}, canal {result['channel']} "
        f"(valid_length={result['valid_length']})"
    )
    print(f"  clean:       mean={result['clean_mean']:.4f}  std={result['clean_std']:.4f}")
    print(
        f"  transformed: mean={result['transformed_mean']:.2e}  std={result['transformed_std']:.6f}"
        f"  (mean_ok={result['mean_ok']}, std_ok={result['std_ok']})"
    )
    print(f"  padding fuera de valid_length es 0: {result['padding_is_zero']}")


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------

def plot_sample_channel(movement_id: int, sample_index: int, sensor_id: int, channel: int | str,
                         stages: tuple[str, ...] = ("clean", "transformed"),
                         manifest_df: pd.DataFrame | None = None, data_dir: str = DEFAULT_DATA_DIR,
                         ax=None):
    """Grafica, una al lado de la otra, la senial de una muestra/canal en las
    etapas pedidas (por defecto clean vs. transformed), leyendo directo de los
    .npy ya guardados."""
    if manifest_df is None:
        manifest_df = load_manifest(data_dir)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, len(stages), figsize=(5.5 * len(stages), 4), squeeze=False)
        ax = ax[0]

    ch_name = resolve_channel(sensor_id, channel)[1]
    movement_name = u.MOVEMENT_LIBRARY[movement_id][0]
    row = get_manifest_row(manifest_df, movement_id, sample_index)

    for axi, stage in zip(ax, stages):
        sig, vlen, _ = get_sample_channel(
            stage, movement_id, sample_index, sensor_id, channel, manifest_df, data_dir
        )
        array_row = row_index_for_stage(stage, row, sample_index)
        row_label = f"fila raw {array_row}" if stage in RAW_ALIGNED_STAGES else f"muestra {sample_index}"
        axi.plot(sig)
        axi.set_title(f"{stage}\nmov {movement_id} ({movement_name}), {row_label}, {ch_name}")
        axi.set_xlabel("timestep")
        axi.set_ylabel("grados" if stage != "transformed" else "z-score")

    if fig is not None:
        fig.tight_layout()
    return ax


def plot_sample_all_channels(stage: str, movement_id: int, sample_index: int, sensor_id: int,
                              manifest_df: pd.DataFrame | None = None, data_dir: str = DEFAULT_DATA_DIR):
    """Grid 2x3 con los 6 canales de un sensor para una muestra, en una sola etapa.
    Util para revisar visualmente una muestra completa de un vistazo."""
    if manifest_df is None:
        manifest_df = load_manifest(data_dir)

    channels = u.channels_for(sensor_id)
    movement_name = u.MOVEMENT_LIBRARY[movement_id][0]
    row = get_manifest_row(manifest_df, movement_id, sample_index)
    array_row = row_index_for_stage(stage, row, sample_index)
    row_label = f"fila raw {array_row}" if stage in RAW_ALIGNED_STAGES else f"muestra {sample_index}"

    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ch_idx, axi in zip(range(len(channels)), axes.ravel()):
        sig, vlen, ch_name = get_sample_channel(
            stage, movement_id, sample_index, sensor_id, ch_idx, manifest_df, data_dir
        )
        axi.plot(sig)
        axi.set_title(ch_name)
        axi.set_xlabel("timestep")

    fig.suptitle(f"{stage}: mov {movement_id} ({movement_name}), sensor {sensor_id}, {row_label}")
    fig.tight_layout()
    return axes
