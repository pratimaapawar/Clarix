import io
import math
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# STARTUP: load the 8 PaySim artifacts once from ../models
# --------------------------------------------------------------------------- #
MODELS_DIR = "../models"

_current_file = None  # tracks which artifact is loading, for clear error messages
try:
    _current_file = "scaler.pkl"
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")

    _current_file = "isolation_forest.pkl"
    isolation_forest = joblib.load(f"{MODELS_DIR}/isolation_forest.pkl")

    _current_file = "xgboost_model.pkl"
    xgboost_model = joblib.load(f"{MODELS_DIR}/xgboost_model.pkl")

    _current_file = "lightgbm_model.pkl"
    lightgbm_model = joblib.load(f"{MODELS_DIR}/lightgbm_model.pkl")

    _current_file = "meta_model.pkl"
    meta_model = joblib.load(f"{MODELS_DIR}/meta_model.pkl")

    _current_file = "calibrated_model.pkl"
    calibrated_model = joblib.load(f"{MODELS_DIR}/calibrated_model.pkl")

    _current_file = "feature_cols.pkl"
    feature_cols = list(joblib.load(f"{MODELS_DIR}/feature_cols.pkl"))

    _current_file = "threshold.pkl"
    threshold = float(joblib.load(f"{MODELS_DIR}/threshold.pkl"))
except Exception as e:
    raise RuntimeError(
        f"Failed to load model artifact '{_current_file}' from '{MODELS_DIR}': {e}"
    )

# Build the SHAP explainer on the XGBoost model once at startup.
explainer = shap.TreeExplainer(xgboost_model)

app = FastAPI(title="Clarix Fraud API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# PYDANTIC MODEL — flexible dict so the same shape works for form + samples
# --------------------------------------------------------------------------- #
class Transaction(BaseModel):
    features: Dict[str, Any]


# --------------------------------------------------------------------------- #
# FEATURE ENGINEERING (must match the training notebook exactly)
# --------------------------------------------------------------------------- #
def _num(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, float) and math.isnan(v):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# TRANSACTION TYPES
# --------------------------------------------------------------------------- #
# The model was trained on PaySim TRANSFER and CASH_OUT only.
#   CASH_IN — cash deposit: the balance goes UP, which the model never saw, so
#             it is judged by the deterministic balance rules alone.
MODEL_TYPES = {"TRANSFER", "CASH_OUT"}
RULE_ONLY_TYPES = {"CASH_IN"}
SUPPORTED_TYPES = MODEL_TYPES | RULE_ONLY_TYPES
NO_DEST_TYPES = {"CASH_OUT", "CASH_IN"}


def _type(d: Dict[str, Any]) -> str:
    return str(d.get("type", "")).strip().upper()


def _build_row(d: Dict[str, Any]) -> Dict[str, float]:
    """Compute the 12 PaySim features from one raw transaction dict."""
    typ = _type(d)
    amount = _num(d.get("amount"))
    oldO = _num(d.get("oldbalanceOrg"))
    newO = _num(d.get("newbalanceOrig"))
    oldD = _num(d.get("oldbalanceDest"))
    newD = _num(d.get("newbalanceDest"))
    step = _num(d.get("step"))

    return {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldO,
        "newbalanceOrig": newO,
        "oldbalanceDest": oldD,
        "newbalanceDest": newD,
        "type_TRANSFER": 1 if typ == "TRANSFER" else 0,
        "errorBalanceOrig": newO + amount - oldO,
        "errorBalanceDest": oldD + amount - newD,
        "origEmptied": 1 if (oldO > 0 and newO == 0) else 0,
        "amountToBalanceRatio": amount / (oldO + 1),
        "amountDevFromOrigMean": 0.0,  # single transaction has no account history
    }


def build_features(d: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([_build_row(d)]).reindex(columns=feature_cols)


def build_features_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([_build_row(d) for d in rows]).reindex(columns=feature_cols)


def apply_rules(d: Dict[str, Any]) -> List[str]:
    """Pure-arithmetic fraud checks — run before/alongside the model, no ML.

    Catches deterministic red flags (e.g. balances that don't reconcile) that a
    statistical model can miss when the pattern is outside its training data.
    """
    typ = _type(d)
    amount = _num(d.get("amount"))
    oldO = _num(d.get("oldbalanceOrg"))
    newO = _num(d.get("newbalanceOrig"))
    oldD = _num(d.get("oldbalanceDest"))
    newD = _num(d.get("newbalanceDest"))

    rules_fired: List[str] = []

    if typ == "CASH_IN":
        # Deposit: the account balance must rise by exactly the amount.
        if abs((oldO + amount) - newO) > 0.01:
            rules_fired.append("Deposit not reflected correctly in account balance")
        if newO < oldO:
            rules_fired.append("Balance decreased after a cash deposit")
        return rules_fired

    expected_after = oldO - amount
    if abs(expected_after - newO) > 0.01:
        rules_fired.append("Balance math doesn't add up — funds unaccounted for")
    if oldO > 0 and newO == 0:
        rules_fired.append("Account emptied completely")
    if typ not in NO_DEST_TYPES and newD == oldD and amount > 0:
        rules_fired.append("Destination balance didn't increase despite transfer")
    if amount > oldO:
        rules_fired.append("Transfer exceeds available balance")
    return rules_fired


def risk_band(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    if prob >= 0.40:
        return "MEDIUM"
    return "LOW"


def score_matrix(X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Run the full stacking chain on a feature DataFrame (1+ rows)."""
    Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    iso = -isolation_forest.score_samples(Xs)
    xp = xgboost_model.predict_proba(Xs)[:, 1]
    lp = lightgbm_model.predict_proba(Xs)[:, 1]
    meta_in = np.column_stack([iso, xp, lp])
    prob = calibrated_model.predict_proba(meta_in)[:, 1]
    return {"Xs": Xs, "iso": iso, "xgb": xp, "lgb": lp, "prob": prob}


def score_transaction(d: Dict[str, Any]) -> dict:
    X = build_features(d)
    s = score_matrix(X)
    prob = float(s["prob"][0])
    return {
        "prob": prob,
        "is_fraud": bool(prob >= float(threshold)),
        "iso": float(s["iso"][0]),
        "xgb": float(s["xgb"][0]),
        "lgb": float(s["lgb"][0]),
        "X": X,
        "Xs": s["Xs"],
    }


# --------------------------------------------------------------------------- #
# ENDPOINTS
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {"status": "running", "threshold": round(float(threshold), 4)}


@app.get("/examples")
def examples():
    """Ready-to-run demo transactions.

    Fraud in PaySim = the origin account is fully drained to 0 while the
    destination isn't properly credited. Normal = balances add up.
    """
    return {
        "fraud": [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            },
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 90000.0,
                "oldbalanceOrg": 90000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            },
        ],
        "normal": [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 5000.0,
                "oldbalanceOrg": 90000.0,
                "newbalanceOrig": 85000.0,
                "oldbalanceDest": 20000.0,
                "newbalanceDest": 25000.0,
            },
            {
                "step": 1,
                "type": "CASH_OUT",
                "amount": 1200.0,
                "oldbalanceOrg": 50000.0,
                "newbalanceOrig": 48800.0,
                "oldbalanceDest": 10000.0,
                "newbalanceDest": 11200.0,
            },
        ],
    }


@app.post("/predict")
def predict(t: Transaction):
    try:
        raw = t.features
        typ = _type(raw)
        if typ not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported type '{typ}'. Use one of: {', '.join(sorted(SUPPORTED_TYPES))}",
            )

        # 1. Rule layer runs first — pure arithmetic, no model needed.
        rules_fired = apply_rules(raw)

        if typ in RULE_ONLY_TYPES:
            # The model was never trained on deposits — verdict is rules only.
            X = build_features(raw)
            return {
                "is_fraud": bool(rules_fired),
                "risk_score": 1.0 if rules_fired else 0.0,
                "risk_tier": "HIGH" if rules_fired else "LOW",
                "rules_fired": rules_fired,
                "flagged_by": {"rules": bool(rules_fired), "model": False},
                "scored_by": "rules",
                "note": f"{typ} is not covered by the ML model (trained on TRANSFER / "
                        "CASH_OUT), so this verdict comes from the balance rule checks.",
                "explanation": [],
                "model_scores": {},
                "features": {name: float(X.iloc[0][name]) for name in feature_cols},
                "transaction": raw,
            }

        # 2. Model runs as before: scale -> iso -> xgb -> lgb -> stack -> calibrate.
        result = score_transaction(raw)
        prob = result["prob"]
        Xs = result["Xs"]
        X = result["X"]

        # 3. Combine: fraud if any rule fired OR model score >= threshold.
        model_fraud = bool(prob >= float(threshold))
        is_fraud = bool(rules_fired) or model_fraud

        # A fired rule is a hard, deterministic fraud signal → treat as full
        # confidence so the displayed gauge matches the verdict. The raw model
        # probability is still returned untouched in model_scores.final.
        rule_score = 1.0 if rules_fired else 0.0
        combined_score = max(prob, rule_score)

        # SHAP explanation for the single row.
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[-1]
        sv_row = np.asarray(sv)[0]

        drivers = [
            {
                "feature": name,
                "effect": "raised risk" if val > 0 else "lowered risk",
                "impact": round(float(val), 4),
            }
            for name, val in zip(feature_cols, sv_row)
        ]
        drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
        top5 = drivers[:5]

        # Tier reflects the combined verdict: a fired rule is a hard HIGH.
        tier = "HIGH" if rules_fired else risk_band(prob)

        features_out = {name: float(X.iloc[0][name]) for name in feature_cols}

        return {
            "is_fraud": is_fraud,
            "risk_score": round(combined_score, 4),
            "risk_tier": tier,
            "rules_fired": rules_fired,
            "flagged_by": {
                "rules": bool(rules_fired),
                "model": model_fraud,
            },
            "scored_by": "model+rules",
            "explanation": top5,
            "model_scores": {
                "isolation_forest": round(result["iso"], 4),
                "xgboost": round(result["xgb"], 4),
                "lightgbm": round(result["lgb"], 4),
                "final": round(prob, 4),
            },
            "features": features_out,
            "transaction": raw,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    required = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required column(s): {', '.join(missing)}",
        )

    # Score TRANSFER / CASH_OUT with the model and CASH_IN with the rules;
    # other PaySim types (PAYMENT, DEBIT) are skipped.
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    subset = df[df["type"].isin(SUPPORTED_TYPES)].copy()

    if subset.empty:
        return {
            "summary": {"total": 0, "flagged": 0, "flag_rate": 0},
            "results": [],
        }

    records = subset.to_dict("records")
    X = build_features_frame(records)
    s = score_matrix(X)
    probs = s["prob"]

    results = []
    for i, rec in enumerate(records):
        if rec["type"] in RULE_ONLY_TYPES:
            p = 1.0 if apply_rules(rec) else 0.0
        else:
            p = float(probs[i])
        results.append(
            {
                "index": int(i),
                "type": str(rec.get("type", "")),
                "amount": _num(rec.get("amount")),
                "risk_score": round(p, 4),
                "risk_tier": risk_band(p),
                "is_fraud": bool(p >= float(threshold)),
            }
        )

    results.sort(key=lambda r: r["risk_score"], reverse=True)

    total = len(results)
    flagged = sum(1 for r in results if r["is_fraud"])
    flag_rate = round(flagged / total * 100, 2) if total else 0
    summary = {"total": total, "flagged": flagged, "flag_rate": flag_rate}

    return {"summary": summary, "results": results}


# Run:  cd backend  &&  uvicorn app:app --reload --port 8000
# Test: open http://localhost:8000/docs
