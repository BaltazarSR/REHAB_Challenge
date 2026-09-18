# REHAB-RETO

Classifying rehabilitation exercises for post-stroke patients from wearable sensor data (a limb IMU + an instrumented glove), following the full ML lifecycle: **Data → Model → Evaluate → Refine → Solution**.

Each repetition is a two-sensor time series, and the task is to identify which of 16 standardized rehab exercises it is. See [`Final_delivery/`](Final_delivery/) for the final model and app; the folders below document how it was built, step by step.

## Project structure

| Folder | Purpose |
|---|---|
| [`Data/`](Data/) | The REHAB dataset: raw recordings plus every derived stage (processed, cleaned, transformed), `manifest.csv` (per-sample metadata, including real-signal length), and dataset documentation/citation. |
| [`ETL_Cleaning/`](ETL_Cleaning/) | The cleaning & transformation pipeline (spike repair, smoothing, per-sample normalization) that turns raw data into `Data/Rehab_exercise/d03_clean_data` and `d04_transformed_data`. |
| [`EDA_Report/`](EDA_Report/) | Exploratory data analysis: class balance, distributions, channel correlations, and the data-quality issues found (and why they matter for modeling). |
| [`Model_from_scratch/`](Model_from_scratch/) | A multinomial logistic regression (softmax regression) implemented from scratch (no sklearn), trained on hand-engineered per-segment statistical features. |
| [`Model_selection/`](Model_selection/) | Compares three sequence-native architectures, a two-branch 1D-CNN, LSTM, and Transformer encoder, to pick the best one to carry forward. |
| [`Final_delivery/`](Final_delivery/) | **The final deliverable.** The winning Transformer architecture (`Transformer.py`), its full configuration/training/validation/regularization/hyperparameter-tuning process (`model_training.ipynb`), the trained weights (`transformer_model.pt`), and a Streamlit app (`main.py`) to try the model interactively. |
| [`utils/`](utils/) | Shared utility functions (ETL helpers, manifest/data loading, inspection & plotting) reused across the notebooks above. |
| [`sandbox/`](sandbox/) | Scratch notebooks for exploring the utility functions; not part of the main pipeline. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running things

- **ETL pipeline:** `jupyter nbconvert --to notebook --execute --inplace ETL_Cleaning/ETL_Cleaning.ipynb`
- **EDA:** open `EDA_Report/Activity1.ipynb` and `EDA_Report/EDA_Graphs.ipynb`
- **Model selection (CNN vs LSTM vs Transformer):** open `Model_selection/model_selection.ipynb`
- **Final training pipeline:** open `Final_delivery/model_training.ipynb`
- **Streamlit app:** `streamlit run Final_delivery/main.py`

## Running the Final Delivery

The final deliverable lives in `Final_delivery/`: the trained Transformer (`Transformer.py` + `transformer_model.pt`), the notebook that produced it (`model_training.ipynb`), and a Streamlit app (`main.py`) to try it interactively.

1. Set up the virtual environment.

2. Launch the app from the repo root:
   ```bash
   source .venv/bin/activate
   streamlit run Final_delivery/main.py
   ```
   
3. It opens at `http://localhost:8501` with two sections in the sidebar:
   - **Try the Model** — pick a recorded repetition (or hit "Random sample") and classify it with the trained Transformer.
   - **About the Project** — problem definition, approach, and results.

The trained weights (`Final_delivery/transformer_model.pt`) are already committed, so the app works out of the box with no retraining needed. To regenerate them from scratch instead, run `Final_delivery/model_training.ipynb` end to end (~10 minutes on CPU) — it re-runs the hyperparameter search and overwrites `transformer_model.pt`.

## Dataset

Lv, M. *et al.* (2026). *A wearable sensor–based kinematic dataset collected under standardized rehabilitation tasks from 120 post-stroke patients.* Scientific Data 13:1136. https://doi.org/10.1038/s41597-026-07802-2

## Teacher corrections

1. Always explain the problem and data assuming the "public" is not familiar with it.

2. Use seaborn for graphs instead of matplotlib
