# Clarix — Explainable Fraud Detection

Clarix is a fraud-detection web app for PaySim-style money-transfer data. It
serves a **pre-trained stacking ensemble** (Isolation Forest + XGBoost +
LightGBM, combined by a calibrated meta-model) through a FastAPI backend and a
Streamlit frontend. Every prediction comes with a **SHAP explanation** so you
can see *why* a transaction was flagged.

No training happens in this app — the 8 model artifacts are loaded once and
served as-is.

## Folder structure

```
clarix/
├── models/                    # your 8 pre-trained .pkl files (not modified)
│   ├── scaler.pkl
│   ├── isolation_forest.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── meta_model.pkl
│   ├── calibrated_model.pkl
│   ├── feature_cols.pkl       # list of feature names, in order
│   └── threshold.pkl          # float fraud decision cutoff
├── backend/
│   └── app.py                 # FastAPI service
├── frontend/
│   └── app.py                 # Streamlit UI
├── requirements.txt
└── README.md
```

## Features engineered per transaction

The backend reproduces the training feature pipeline exactly (12 features,
reordered to `feature_cols.pkl`):

`step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
type_TRANSFER, errorBalanceOrig, errorBalanceDest, origEmptied,
amountToBalanceRatio, amountDevFromOrigMean`

## Install

```
pip install -r requirements.txt
```

Make sure your 8 `.pkl` files are in `clarix/models/` before starting.

## Run the backend

```
cd backend
uvicorn app:app --reload --port 8000
```

The API is then available at http://localhost:8000
(`GET /` returns status + threshold; `POST /predict`, `POST /predict-batch`,
`GET /examples`).

## Run the frontend

In a second terminal:

```
cd frontend
streamlit run app.py
```

The UI opens in your browser and connects to the backend automatically. A
green dot next to the title means the backend is reachable; red means it is not.

## Demo tip

On the **Single Transaction** tab, click **Load Fraud Example** and press
**Analyze** — you'll get a red "FRAUD DETECTED" banner, a high risk gauge, and
the SHAP factors explaining the flag. **Load Normal Example** shows a clean
legitimate transaction for contrast. For the **Bulk Upload** tab, drop in a
PaySim CSV to score every TRANSFER / CASH_OUT row at once.
