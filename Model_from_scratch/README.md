# Movement classifier from scratch

Goal: classify each repetition into one of the 16 `movement_id` classes using hand-engineered statistical features + a **multinomial logistic regression** (softmax regression) implemented from scratch. Input data is `Data/Rehab_exercise/d04_transformed_data/` (cleaned, smoothed, per-sample z-scored), aligned to `Data/Rehab_exercise/manifest.csv`.

## Windowing plan

**Unit of analysis:** one repetition = 12 channels total — 6 from `sensorID=1` (limb IMU) + 6 from `sensorID=2` (glove) — each stored as a separate `(880, 6)` array. Only `[0, valid_length)` is real signal; the rest is padding. Per `ETL_Cleaning/README.md`, padding must **never** be inferred by scanning for zeros in `d04` (normalized real values legitimately cross zero too) — always use `valid_length_sensor1` / `valid_length_sensor2` from the manifest.

**Key fact that shapes the design:** `valid_length_sensor1` and `valid_length_sensor2` are tracked *separately* and can diverge a lot for the same repetition — up to 875 timesteps apart in this dataset. So sensor1 and sensor2 must be segmented independently, each using its own `valid_length`. Forcing a shared length (e.g. `min` of the two) would either truncate real limb-sensor signal or stretch degenerate glove segments to fill space that isn't there.

**Method: fixed segment *count*, not fixed segment *width*.**

- Segment count `N` is the same for every sample (so every sample produces the same number of feature columns).

- For sensor `s` in a given sample: `segment_length = valid_length_s / N` (**true division**, a float — not `//`, which would leave a remainder dumped onto the last segment).

- Segment `i` (`i = 0 .. N-1`) spans `[round(i * segment_length), round((i+1) * segment_length))`. Boundaries are computed straight from the float, then rounded to the nearest integer index — this covers `[0, valid_length_s)` exactly, with any rounding slack (at most ±1 timestep) spread across segments instead of piled onto one.

- A fixed *width* window was rejected because `valid_length` ranges from 301 to 880 (sensor1) and 5 to 880 (sensor2) — a fixed width would produce a different segment count per sample, which breaks a fixed-length feature vector.

**Choosing N = 10**, checked against the actual manifest distribution:

- `valid_length_sensor1` never drops below 301 → worst case ~30 timesteps/segment, plenty for a stable median/std/min/max.

- `valid_length_sensor2` (glove) is the binding constraint: only **4 of 3,884 samples** — all `movement_id=13` (ankle ext./knee rotation, a lower-limb movement where the glove barely moves) — have `valid_length_sensor2 < 30`, which would leave some segments with too few points for a meaningful std.

- Decision: **drop those 4 samples** (both `_1` and `_2` arrays, to keep sensor alignment) rather than adding special-case logic for 0.1% of the data. Log the dropped `(movement_id, sample_index)` pairs when this runs.

- Resulting feature count: 12 channels × 6 stats × 10 segments = **720 features/sample**, against ~3,880 samples — reasonable for logistic regression, with room to prune later (the EDA already flagged `f2`–`f5` as 0.87–0.94 correlated).

**Per-segment statistics (6 per channel per segment):** median, mean, std, min, max, skew.

**Column naming: `seg{ii}_{channel}_{stat}`** — e.g. `seg00_pitch1_median`, `seg09_f5_skew`.

- `ii` is the segment index, zero-padded to 2 digits (`00`–`09` at `N=10`) so columns sort lexicographically in the same order as numerically — future-proofs against `N > 9` without a rename.

- `channel` is the real channel name (`pitch1, yaw1, roll1, pitch2, yaw2, roll2` for sensor1; `f1, f2, f3, f4, f5, pitch3` for sensor2), not a generic `channel_1`/`channel_2` index. Since none of the 12 names collide across the two sensors, this doubles as sensor disambiguation for free — no separate `sensor1_`/`sensor2_` prefix needed.

- Single `_` delimiter throughout (no mixed `_`/`-`) — safe because none of `ii`, `channel`, or `stat` values contain an underscore themselves.

**Edge-case rules to encode explicitly:**

- **Frozen segments** (all points identical, e.g. a wireless dropout holding a stale reading — see `ETL_Cleaning/README.md`) do occur: `std == 0` for ~3.3% of segment/channel/sample combinations. `skew` guards this case (`0.0` instead of a `0/0` division), since skew is undefined for a constant segment. This is the only such guard in the code — checked empirically and kept because it's real, not speculative.
- No guard for empty/single-point segments: verified the minimum segment length across the whole dataset (after the `MIN_VALID_LENGTH` drop rule above) is 3, so that case can't occur at `N=10` and isn't coded for.
- No guard for a zero-std (constant) feature *column* in `standardize()` either: verified 0 of the 720 columns are constant across the whole dataset. If a future change to segmenting/stats ever produces one, standardizing would divide by 0 there — worth re-checking if `N_SEGMENTS` or `STATS` change.

Sensor1 and sensor2 segments are computed independently and simply concatenated into one row per sample — segment `i` of sensor1 and segment `i` of sensor2 are *not* claimed to be the same time window, since their `valid_length`s differ.

## Model

Implemented in `main.py`. Multinomial logistic regression (softmax regression), trained from scratch — no sklearn — on the 720-column feature table against `movement_id`.

**Architecture:** `Z = X @ W + b` (`W`: 720 × 16, `b`: 16), `P = softmax(Z)` row-wise, prediction = `argmax(P)`. `softmax()` subtracts each row's max before exponentiating (`Z - Z.max(axis=1, keepdims=True)`).

**Training:** full-batch gradient descent.

- `grad_W = X.T @ (P - Y) / n_samples`, `grad_b = mean(P - Y, axis=0)`.

- `LEARNING_RATE = 0.05`, `N_ITERATIONS = 2000`, fixed for every run.

**Prediction:** `predict(X)` runs the same forward pass as training (`softmax(X @ W + b)`) then `argmax` to pick the single most likely class.

- **Train/test split** (80/20): a plain random shuffle-split over all samples together.