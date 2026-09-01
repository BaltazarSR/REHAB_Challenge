# ETL_Cleaning — Limpieza y Transformación de `Rehab_exercise`

Este directorio contiene el pipeline de ETL (Extract–Transform–Load) usado para
limpiar y transformar los datos crudos de `Data/Rehab_exercise/d01_raw_data/`.

## Archivos

- **`../utils/etl_utils.py`** (raíz del repo) — funciones reutilizables de
  limpieza y transformación (carga, detección de muestras vacías/duplicadas,
  reparación de valores angulares imposibles, suavizado, normalización).
  Están documentadas con docstrings en español que explican el porqué de cada
  función, no solo el qué.
- **`ETL_Cleaning.ipynb`** — notebook ejecutado de punta a punta que:
  1. Carga los datos crudos.
  2. Audita su calidad (conteos reales, no supuestos).
  3. Limpia, documentando cada decisión en celdas markdown junto con la
     evidencia numérica que la motiva.
  4. Transforma, documentando cada decisión de la misma forma.
  5. Guarda los resultados.

`etl_utils.py` vive en `utils/` en la raíz del repo (no en este directorio)
para poder importarse igual desde `ETL_Cleaning/` y desde `EDA_Report/` — ej.
`from utils import etl_utils as u`.

**Toda la explicación detallada de cada decisión vive en las celdas markdown
del notebook** (Secciones 2–6). Este README es un índice/resumen para
ubicarlas rápido sin tener que abrir el notebook completo.

## Qué se generó

Ejecutar el notebook produce, dentro de `Data/Rehab_exercise/`:

| Carpeta/archivo | Contenido | Unidades |
|---|---|---|
| `d03_clean_data/<movementID>_<sensorID>.npy` | Datos limpios: sin muestras 100% vacías, sin duplicados exactos, con los valores angulares matemáticamente imposibles reparados | grados (igual que el crudo) |
| `d04_transformed_data/<movementID>_<sensorID>.npy` | Datos limpios + suavizados (media móvil, ventana=10) + normalizados (z-score por muestra y canal) | adimensional (z-score) |
| `manifest.csv` | Una fila por muestra conservada: `movement_id`, `movement_name`, `limb`, `sample_index`, `orig_index_in_raw`, `valid_length_sensor1`, `valid_length_sensor2` | — |

`d01_raw_data` y `d02_processed_data` **no se modifican** — son los datos
originales del dataset REHAB tal como se entregan. `d03`/`d04` son productos
derivados de este pipeline.

**Uso del manifest:** es la fuente de verdad de qué parte de cada muestra es
señal real (`[0, valid_length)`) y cuál es relleno (padding). No se debe
inferir el padding buscando ceros en `d04_transformed_data`, porque después de
normalizar los valores reales también pueden cruzar por cero de forma
legítima.

## Resumen de decisiones (detalle completo en el notebook)

### Limpieza

| Problema encontrado | Decisión | Evidencia |
|---|---|---|
| 0 `NaN` / 0 `Inf` en 9,232 muestras crudas | No imputar nada (se deja un chequeo defensivo en `etl_utils.check_nan_inf`) | Auditoría exhaustiva de los 32 archivos |
| 142 muestras 100% relleno (ninguna medición real) | Eliminar la muestra completa, en ambos archivos (`_1` y `_2`) del movimiento, para no romper el alineamiento por índice | `etl_utils.find_empty_samples` sobre `compute_trailing_zero_length` |
| 1,438 renglones duplicados bit-a-bit | Eliminar todos menos la primera ocurrencia (conjunto entre `_1`/`_2`) | `etl_utils.find_duplicate_sample_mask` (comparación exacta con `np.unique`) — una coincidencia bit-a-bit en una señal analógica de 880×6 puntos no ocurre por azar; es un artefacto del pipeline de exportación |
| Relleno parcial restante, de longitud distinta entre `sensorID=1` y `sensorID=2` para el mismo ensayo | No recortar: calcular y persistir `valid_length` por muestra y por sensor en el manifest | Padding promedio observado entre 16% y 86% de las muestras según el movimiento |
| `yaw1, roll1, yaw2, roll2, pitch3` con valores de hasta ~1178° (matemáticamente imposibles: se calculan con `arctan2`, rango exacto `(-180°,180°]`) | Detectar por el límite matemático `|valor| > 180°` (no un umbral estadístico ajustable) y reparar por interpolación lineal dentro de la región válida, sin descartar el ensayo completo | 47,806 puntos individuales reparados en canales de `sensorID=1`, 674 en `pitch3`. Caso documentado: movimiento 4, muestra 124, canal `yaw2` — dentro de esta única muestra hay **3 tramos donde el valor es simultáneamente inválido (fuera de ±180°) Y queda bit a bit idéntico** durante varios timesteps seguidos: -225.03° (3 timesteps), 884.44° (12 timesteps) y 381.83° (3 timesteps). La muestra también tiene otros 7 tramos congelados en valores que sí son válidos (ej. -36.46° durante 45 timesteps) — esos no se consideran evidencia de falla, ya que un valor constante dentro de rango es compatible con el paciente sosteniendo la posición. Un valor repetido bit a bit durante cientos de milisegundos no proviene de un sensor analógico en movimiento; los tramos *inválidos y congelados* son consistentes con una caída de la transmisión inalámbrica (ZigBee) que mantiene el último dato recibido, no con un movimiento real ni con una velocidad angular calculable (no se sabe cuánto tiempo real pasó mientras estaban congelados, así que **no se afirma ninguna velocidad angular específica**). La causa exacta no está confirmada — es plausible que contribuya una inestabilidad tipo *gimbal lock* (pitch1/pitch2 llegan a 89.7° en otras partes del dataset), pero no se confirmó como la causa de este caso puntual. Lo único seguro, sin importar la causa: esos 3 valores almacenados no pueden ser una salida válida de `arctan2` |
| `pitch1, pitch2` (rango `[-90°,90°]`, fórmula `arcsin`) y `f1`–`f5` (rango `[0°,~117°]`, dentro de especificación del guante) | Ninguna corrección | Verificado: 0 violaciones de rango en toda la auditoría |

### Transformación

| Decisión | Por qué |
|---|---|
| Sin conversión de unidades | Los 12 canales ya están en grados de forma consistente (verificado contra `Article.pdf`, Tablas 8–10) |
| Media móvil, ventana=10, calculada solo sobre `[0, valid_length)` | Misma ventana que usa el pipeline original del dataset; limitarla a la región válida evita que el 0 del padding sesgue los últimos puntos reales de la señal |
| Normalización zero-mean/unit-variance por muestra y canal, calculada solo sobre `[0, valid_length)` | Remueve el offset de calibración/colocación específico de cada sensor y paciente (igual que el artículo original); se demostró numéricamente que calcular sobre las 880 posiciones completas subestima la media y la desviación estándar reales (ejemplo en el notebook: media real −14.24 vs. −4.90 incluyendo padding, en una muestra con 65% de relleno) |
| Orden: limpiar (reparar ángulos imposibles) **antes de** transformar | Normalizar antes de reparar dejaría que un valor corrupto de ~887° dominara la media/desviación de toda la muestra, distorsionando también los puntos válidos |

## Cómo reproducir

```bash
cd REHAB-RETO
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib jupyter nbformat nbconvert ipykernel
cd ETL_Cleaning
jupyter nbconvert --to notebook --execute --inplace ETL_Cleaning.ipynb
```

El notebook lee de `../Data/Rehab_exercise/d01_raw_data/` y escribe en
`../Data/Rehab_exercise/{d03_clean_data,d04_transformed_data,manifest.csv}`.
