"""
Funciones reutilizables para el pipeline de ETL (Extract-Transform-Load) del
dataset REHAB - Rehab_exercise.

Estas funciones son usadas por ETL_Cleaning.ipynb. Se separan del notebook
para que el codigo de limpieza/transformacion quede versionado como script
(no solo como celdas de notebook) y sea reutilizable en otros analisis.

Contexto de los datos: ver Data/README.md
"""

from __future__ import annotations

import glob
import os

import numpy as np

# ---------------------------------------------------------------------------
# Metadatos del dataset (extraidos del articulo, ver Data/README.md)
# ---------------------------------------------------------------------------

CHANNELS_SENSOR1 = ["pitch1", "yaw1", "roll1", "pitch2", "yaw2", "roll2"]
CHANNELS_SENSOR2 = ["f1", "f2", "f3", "f4", "f5", "pitch3"]

# Canales derivados con formulas atan2
SPIKE_CHANNELS_SENSOR1 = ["yaw1", "roll1", "yaw2", "roll2"]
SPIKE_CHANNELS_SENSOR2 = ["pitch3"]

MOVEMENT_LIBRARY = {
    0: ("Bobath handshake", "upper"),
    1: ("Bobath flexion/extension", "upper"),
    2: ("Bobath forward flexion/extension", "upper"),
    3: ("Bobath anterior/posterior rotation", "upper"),
    4: ("Elbow flexion and wrist compression", "upper"),
    5: ("Wrist flexion and extension", "upper"),
    6: ("Finger-to-finger training", "upper"),
    7: ("Ball gripping", "upper"),
    8: ("Shoulder joint internal and external rotation", "upper"),
    9: ("Breast expansion", "upper"),
    10: ("Flexion-pressure rotation forward and backward", "upper"),
    11: ("Elbow joint flexion and touch", "upper"),
    12: ("Shoulder touch training", "upper"),
    13: ("Ankle extension & Knee internal/external rotation", "lower"),
    14: ("Knee flexion and extension", "lower"),
    15: ("Hip flexion and extension", "lower"),
}

SIGNAL_LENGTH = 880
N_MOVEMENTS = 16

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def list_raw_files(raw_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(raw_dir, "*.npy")))


def load_movement(raw_dir: str, movement_id: int, sensor_id: int) -> np.ndarray:
    """Carga el archivo <movementID>_<sensorID>.npy con padding de ceros en el nombre."""
    fname = f"{movement_id:03d}_{sensor_id}.npy"
    path = os.path.join(raw_dir, fname)
    return np.load(path)


def channels_for(sensor_id: int) -> list[str]:
    return CHANNELS_SENSOR1 if sensor_id == 1 else CHANNELS_SENSOR2


# ---------------------------------------------------------------------------
# Clean: deteccion de estructura / calidad
# ---------------------------------------------------------------------------

def check_nan_inf(a: np.ndarray) -> dict:
    """Cuenta valores NaN/Inf. En la auditoria del dataset crudo el resultado
    fue 0 en los 32 archivos; se deja como verificacion defensiva."""
    return {"nan_count": int(np.isnan(a).sum()), "inf_count": int(np.isinf(a).sum())}


def compute_trailing_zero_length(a: np.ndarray) -> np.ndarray:
    """Para cada muestra, cuenta cuantos timesteps consecutivos al final son
    exactamente cero en TODOS los canales simultaneamente. Esto corresponde
    al relleno (zero-padding) que el dataset original aplica al final de la
    señal cuando la señal real es mas corta que 880 puntos.

    No se usa para detectar ceros "interiores", solo se cuenta la
    corrida contigua de ceros que termina exactamente en el ultimo indice.
    """
    d, n, _ = a.shape
    zero_row = np.all(a == 0.0, axis=2)
    reversed_zero = zero_row[:, ::-1]
    # cumprod de booleanos: se vuelve 0 en cuanto aparece un renglon no-cero
    running = np.cumprod(reversed_zero.astype(np.int8), axis=1)
    return running.sum(axis=1)                    # (d,)


def find_empty_samples(trailing_zero_len: np.ndarray, n: int = SIGNAL_LENGTH) -> np.ndarray:
    """Muestras que son 100% padding (ningun timestep real -> grabacion vacia)."""
    return trailing_zero_len == n


def find_duplicate_sample_mask(a: np.ndarray) -> np.ndarray:
    """Regresa mascara booleana (True = descartar) marcando, para cada grupo
    de muestras identicas byte a byte, todas menos la primera ocurrencia.

    Justificacion: una repeticion fisica real, capturada por un sensor
    analogico a 50 Hz durante ~17 segundos, no produce una secuencia de
    880x6 valores de punto flotante identica bit a bit a otra muestra. Una
    coincidencia exacta indica una duplicacion en el pipeline de exportacion
    de datos, no un movimiento real repetido.
    """
    d = a.shape[0]
    flat = a.reshape(d, -1)
    _, first_idx, inverse = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    # inverse[i] = indice del valor unico correspondiente a la fila i
    # keep = primera aparicion de cada patron unico
    keep_original_idx = set(first_idx.tolist())
    drop_mask = np.array([i not in keep_original_idx for i in range(d)])
    return drop_mask


# ---------------------------------------------------------------------------
# Clean: reparacion de valores angulares invalidos en canales rotacionales
# ---------------------------------------------------------------------------

# yaw y roll se calculan con arctan2, cuyo rango matematico 
# es exactamente (-180, 180] grados. pitch3 es el mismo
# tipo de angulo derivado para S5.
VALID_ANGLE_BOUND_DEG = 180.0


def out_of_bounds_mask_2d(x: np.ndarray, valid_len: np.ndarray,
                           bound: float = VALID_ANGLE_BOUND_DEG) -> np.ndarray:
    """Marca como invalidos los puntos cuyo valor absoluto excede `bound`.

    A diferencia de un filtro estadistico (p.ej. Hampel/mediana movil), este
    criterio no depende de un umbral ajustable ni falla ante corrupciones
    sostenidas de varios timesteps consecutivos (verificado en la auditoria:
    algunas fallas de sensor se mantienen "pegadas" en un valor invalido
    durante 10+ timesteps antes de saltar bruscamente de vuelta a un valor
    plausible). El limite de 180 grados es una consecuencia matematica de
    arctan2, no una eleccion de disenio.

    Los puntos fuera de la region valida (padding, timestep >= valid_len)
    nunca se marcan: no tiene sentido "reparar" relleno sintetico.
    """
    d, n = x.shape
    mask = np.abs(x) > bound
    t_idx = np.arange(n)[None, :]
    valid_region = t_idx < valid_len[:, None]
    return mask & valid_region


def repair_spikes_1d(row: np.ndarray, mask: np.ndarray, valid_len: int) -> np.ndarray:
    """Interpola linealmente los puntos marcados como pico usando unicamente
    los puntos limpios dentro de la region valida [0, valid_len)."""
    row = row.copy()
    if valid_len < 2 or not mask[:valid_len].any():
        return row
    idx = np.arange(valid_len)
    good = ~mask[:valid_len]
    if good.sum() < 2:
        return row  # no hay suficientes puntos limpios para interpolar
    row[:valid_len] = np.interp(idx, idx[good], row[:valid_len][good])
    return row


def repair_spike_channel(x: np.ndarray, valid_len: np.ndarray,
                          bound: float = VALID_ANGLE_BOUND_DEG) -> tuple[np.ndarray, np.ndarray]:
    """Aplica deteccion (limite matematico +-180) + reparacion por
    interpolacion lineal a un canal (d, n). Regresa (canal_reparado,
    mascara_de_puntos_invalidos_detectados)."""
    mask = out_of_bounds_mask_2d(x, valid_len, bound=bound)
    out = x.copy()
    rows_with_spikes = np.where(mask.any(axis=1))[0]
    for i in rows_with_spikes:
        out[i] = repair_spikes_1d(x[i], mask[i], int(valid_len[i]))
    return out, mask


# ---------------------------------------------------------------------------
# Transform: filtrado y normalizacion respetando la region valida
# ---------------------------------------------------------------------------

def moving_average_valid(a: np.ndarray, valid_len: np.ndarray, window: int = 10) -> np.ndarray:
    """Filtro de media movil (misma ventana que usa el articulo original,
    window=10) aplicado SOLO dentro de la region valida [0, valid_len) de
    cada muestra. El padding se deja igual (en 0) para no introducir senal
    falsa alli.

    Si `valid_len` de una muestra es menor que `window`, se usa una ventana
    mas chica (del tamanio de `valid_len`) para esa muestra unicamente --
    evita que un ensayo inusualmente corto (pocos timesteps reales) rompa la
    convolucion; en la practica esto es raro (la mayoria de los ensayos
    tienen cientos de timesteps validos).

    Cerca de los bordes de la region valida (t proximo a 0 o a valid_len-1)
    la ventana de `window` muestras se sale de la señal real. Un
    np.convolve(..., mode="same") normal trata lo que esta fuera de `seg`
    como cero, lo que sesga esos bordes hacia 0 -- el mismo problema que
    zscore_normalize_valid evita para el padding, pero aqui aplicado a
    puntos que SI son señal real. Por eso se divide, en cada posicion, entre
    la cantidad real de muestras que cayeron dentro de la ventana (`counts`,
    via convolucion de un vector de unos), no siempre entre `w`: la ventana
    se encoge en los bordes en vez de promediar con ceros inventados.
    """
    d, n, c = a.shape
    out = a.copy()
    for i in range(d):
        vlen = int(valid_len[i])
        if vlen < 2:
            continue
        seg = a[i, :vlen, :]
        w = min(window, vlen)
        ones_w = np.ones(w)
        counts = np.convolve(np.ones(vlen), ones_w, mode="same")
        for ch in range(c):
            summed = np.convolve(seg[:, ch], ones_w, mode="same")
            out[i, :vlen, ch] = summed / counts
    return out


def zscore_normalize_valid(a: np.ndarray, valid_len: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalizacion zero-mean/unit-variance POR MUESTRA Y POR CANAL,
    calculada solo sobre la region valida (no sobre el padding).

    Justificacion (ver notebook): el objetivo (igual que en el articulo) es
    remover el offset/drift de calibracion propio de cada colocacion de
    sensor/paciente. Si la media/desviacion se calculan incluyendo el
    padding de ceros, ambas quedan sesgadas hacia 0 -- el sesgo es grande
    para movimientos con mucho padding (hasta 85% de las muestras en el
    dataset). Por eso aqui se calculan exclusivamente sobre [0, valid_len).
    El padding se deja en 0 tras la normalizacion (sigue siendo distinguible
    usando el manifest de valid_length, no por su valor).
    """
    d, n, c = a.shape
    out = np.zeros_like(a)
    for i in range(d):
        vlen = int(valid_len[i])
        if vlen < 2:
            continue
        seg = a[i, :vlen, :]
        mean = seg.mean(axis=0, keepdims=True)
        std = seg.std(axis=0, keepdims=True)
        out[i, :vlen, :] = (seg - mean) / (std + eps)
    return out
