# Clarix — Explainable Financial Fraud Detection

**Clarix** is an explainable fraud-detection application designed to identify suspicious financial transactions using a hybrid machine-learning approach. Built on PaySim-style transaction data, it combines a stacking ensemble with rule-based feature engineering and SHAP explainability to provide both fraud predictions and insights into why a transaction was flagged.

The application serves a **pre-trained ensemble model** through a FastAPI backend and an interactive Streamlit frontend.

## Features

- **Hybrid Fraud Detection:** Combines Isolation Forest, XGBoost and LightGBM with a calibrated stacking/meta-model architecture.
- **Explainable Predictions:** Uses SHAP to highlight the factors influencing individual fraud predictions.
- **Single Transaction Analysis:** Analyze transactions individually and receive fraud verdicts and risk insights.
- **Batch Analysis:** Upload transaction CSV files to analyze multiple transactions.
- **Feature Engineering:** Reproduces the original training feature pipeline for consistent inference.
- **REST API:** Exposes prediction, batch prediction and example endpoints through FastAPI.
- **Pre-trained Models:** Loads saved model artifacts without retraining during application execution.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Deployment | Render |

## Model Architecture

Clarix uses a hybrid fraud-detection pipeline that combines multiple models and a calibrated prediction layer.

- **Isolation Forest:** Detects anomalous transaction patterns.
- **XGBoost:** Identifies fraud patterns using supervised learning.
- **LightGBM:** Provides an additional supervised fraud classification signal.
- **Meta-model / Stacking Layer:** Combines model outputs.
- **Calibration:** Uses the calibrated model and saved decision threshold to support final fraud classification.
- **SHAP:** Provides interpretability for predictions.

The trained model artifacts are loaded at runtime. **No model training takes place inside the deployed application.**

## Features Engineered per Transaction

The backend reproduces the original training feature pipeline, generating 12 features and ordering them according to `feature_cols.pkl`.

| Feature | Description |
|---|---|
| `step` | Transaction time step |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Origin account balance before transaction |
| `newbalanceOrig` | Origin account balance after transaction |
| `oldbalanceDest` | Destination account balance before transaction |
| `newbalanceDest` | Destination account balance after transaction |
| `type_TRANSFER` | Encoded transaction type indicator |
| `errorBalanceOrig` | Origin balance consistency feature |
| `errorBalanceDest` | Destination balance consistency feature |
| `origEmptied` | Indicates whether the origin account was emptied |
| `amountToBalanceRatio` | Transaction amount relative to account balance |
| `amountDevFromOrigMean` | Transaction amount deviation from origin-account mean |

## Project Structure

```text
clarix/
├── models/
│   ├── scaler.pkl
│   ├── isolation_forest.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── meta_model.pkl
│   ├── calibrated_model.pkl
│   ├── feature_cols.pkl
│   └── threshold.pkl
├── backend/
│   └── app.py
├── frontend/
│   └── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd clarix
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Ensure that all eight pre-trained model artifacts are available inside the `models/` directory.

### 3. Run the Backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

The API will be available at:

`http://localhost:8000`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status and decision threshold |
| `/predict` | POST | Analyze a single transaction |
| `/predict-batch` | POST | Analyze multiple transactions |
| `/examples` | GET | Retrieve example transactions |

Interactive API documentation is available through FastAPI Swagger UI:

`http://localhost:8000/docs`

### 4. Run the Frontend

Open a second terminal:

```bash
cd frontend
streamlit run app.py
```

The Streamlit application will open in your browser and connect to the backend API.

## Using Clarix

### Single Transaction Analysis

1. Open the **Single Transaction** tab.
2. Click **Load Fraud Example** or **Load Normal Example**.
3. Click **Analyze**.
4. View the prediction, risk assessment and SHAP explanation.

### Bulk Transaction Analysis

1. Navigate to the **Bulk Upload** tab.
2. Upload a PaySim-compatible CSV file.
3. Run the analysis to evaluate supported transactions.
4. Review the resulting fraud predictions.

## Deployment

Clarix is deployed with a backend API and a web interface, enabling users to access the fraud-detection application without running the entire project locally.

- **Live Application:** [Insert Live Application URL]
- **API:** [Insert Backend API URL]
- **API Documentation:** [Insert Swagger URL]

## Disclaimer

Clarix is an experimental fraud-detection project developed using simulated PaySim transaction data. Predictions are model-generated assessments and should not be treated as definitive evidence of financial fraud.

---

*Developed as an AI & Data Science project focused on machine learning, financial anomaly detection, explainable AI and API-based deployment.*
