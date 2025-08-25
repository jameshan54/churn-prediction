# Streamlit Retention Ops Dashboard
# --------------------------------
# Purpose: Operationalize churn prediction results so Ops/Marketing can act immediately.
# - Ingest latest customer feature snapshot (same schema as your Part 2 df)
# - Score customers with the trained model (XGBoost/any sklearn-compatible .predict_proba)
# - Segment by causes (behavior + sentiment flags)
# - Recommend an action per customer (rule-based starter)
# - Compute ROI per customer using inputs: ARPU, retention window, purchase freq, success rate, cost
# - Create a prioritized target list (treat if ROI>0 AND churn_prob>=threshold AND success_rate>=min_threshold)
# - Export CSV for outreach
#
# How to run:
# 1) Save as app.py
# 2) `pip install streamlit pandas numpy scikit-learn xgboost`
# 3) `streamlit run app.py`
#
# Notes:
# - The app expects a feature table with columns like those in Part 2 (see REQUIRED_FEATURES below).
# - If you upload a pickled pipeline/model, it must implement .predict_proba(X)[:,1].
# - You can also run in "score-less demo" mode by supplying a CSV that already contains `churn_score`.

import io
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Config & Constants
# ----------------------------
st.set_page_config(page_title="Retention Ops Dashboard", layout="wide")

# Feature columns from your Part 2 code. The app will auto-intersect with uploaded data columns.
REQUIRED_FEATURES: List[str] = [
    "total_orders","avg_order_value","avg_num_items","avg_unique_products",
    "avg_actual_delivery_days","tenure_days","order_frequency","monetary",
    "avg_review_score","last_review_score","review_count",
    "complaint_kw_any","complaint_kw_ratio",
    "low_freq_flag","long_delivery_flag","low_star_flag","no_purchase_30d"
]

ID_CANDIDATES = ["customer_unique_id", "customer_id"]

# ----------------------------
# Utility Functions
# ----------------------------

def pick_id_column(df: pd.DataFrame) -> str:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def ensure_no_purchase_30d(df: pd.DataFrame) -> pd.DataFrame:
    # If the flag is absent, derive from recency_days (1 if no purchase in last 30d)
    if "no_purchase_30d" not in df.columns:
        if "recency_days" in df.columns:
            df["no_purchase_30d"] = (df["recency_days"] > 30).astype(int)
        else:
            df["no_purchase_30d"] = 0
    return df


def compute_ltv(arpu_per_order: float, months: float, orders_per_month: float) -> float:
    # Simplified LTV loss avoided if churn is prevented
    return max(arpu_per_order, 0) * max(months, 0) * max(orders_per_month, 0)


def recommend_action(row: pd.Series) -> str:
    # Simple, editable rule-set grounded in your Part 2 flags
    if int(row.get("complaint_kw_any", 0)) == 1 or int(row.get("low_star_flag", 0)) == 1:
        return "CS Callback + Apology Coupon"
    if int(row.get("long_delivery_flag", 0)) == 1:
        return "Shipping Voucher"
    if float(row.get("monetary", 0)) >= np.nanmedian(row.get("monetary", 0)):
        return "VIP Perk"
    return "Coupon 10%"


def success_rate_for_action(action: str, sr_map: Dict[str, float]) -> float:
    return float(sr_map.get(action, 0.3))


def cost_for_action(action: str, cost_map: Dict[str, float]) -> float:
    return float(cost_map.get(action, 0.0))


def compute_roi_row(row: pd.Series, ltv_loss: float, sr_map: Dict[str, float], cost_map: Dict[str, float]) -> Dict[str, float]:
    p_churn = float(row.get("churn_score", 0))
    action = row.get("action", "Coupon 10%")
    p_success = success_rate_for_action(action, sr_map)
    cost = cost_for_action(action, cost_map)
    expected_saved = p_churn * p_success * ltv_loss
    roi = expected_saved - cost
    return {
        "p_churn": p_churn,
        "p_success": p_success,
        "expected_saved": expected_saved,
        "cost": cost,
        "roi": roi,
    }


def score_with_model(model_obj, X: pd.DataFrame) -> np.ndarray:
    # Accept sklearn/xgboost-like fitted models/pipelines exposing predict_proba
    try:
        proba = model_obj.predict_proba(X)[:, 1]
    except Exception as e:
        raise RuntimeError(f"Model scoring failed: {e}")
    return proba


# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Data & Model")
    data_file = st.file_uploader("Upload latest feature snapshot CSV", type=["csv"])
    model_file = st.file_uploader("Upload trained model (.pkl)", type=["pkl"])

    st.caption("The CSV should include columns similar to your Part 2 master table.\n"
               "If you don't upload a model, provide a pre-scored column named `churn_score`.")

    st.subheader("Thresholds")
    risk_threshold = st.slider("Churn score threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    min_success_threshold = st.slider("Min retention success rate threshold", 0.0, 1.0, 0.30, 0.01)

    st.subheader("LTV Inputs")
    arpu = st.number_input("ARPU per order", min_value=0.0, value=100.0, step=1.0)
    retention_months = st.number_input("Retention window (months)", min_value=0.0, value=3.0, step=0.5)
    orders_per_month = st.number_input("Expected orders per month", min_value=0.0, value=1.0, step=0.1)
    ltv_loss = compute_ltv(arpu, retention_months, orders_per_month)
    st.metric("Estimated LTV loss if churned (per user)", f"${ltv_loss:,.2f}")

    st.subheader("Action Success Rates (editable)")
    sr_coupon = st.slider("Coupon 10%", 0.0, 1.0, 0.30, 0.01)
    sr_cs = st.slider("CS Callback + Apology Coupon", 0.0, 1.0, 0.35, 0.01)
    sr_ship = st.slider("Shipping Voucher", 0.0, 1.0, 0.32, 0.01)
    sr_vip = st.slider("VIP Perk", 0.0, 1.0, 0.28, 0.01)
    success_map = {
        "Coupon 10%": sr_coupon,
        "CS Callback + Apology Coupon": sr_cs,
        "Shipping Voucher": sr_ship,
        "VIP Perk": sr_vip,
    }

    st.subheader("Action Costs (editable)")
    cost_coupon = st.number_input("Coupon 10% cost (avg)", min_value=0.0, value=8.0, step=1.0)
    cost_cs = st.number_input("CS Callback + Apology Coupon cost", min_value=0.0, value=10.0, step=1.0)
    cost_ship = st.number_input("Shipping Voucher cost", min_value=0.0, value=7.0, step=1.0)
    cost_vip = st.number_input("VIP Perk cost", min_value=0.0, value=12.0, step=1.0)
    cost_map = {
        "Coupon 10%": cost_coupon,
        "CS Callback + Apology Coupon": cost_cs,
        "Shipping Voucher": cost_ship,
        "VIP Perk": cost_vip,
    }

    st.write("")
    st.caption("Treat if: ROI > 0 AND churn_score ≥ threshold AND success_rate ≥ min threshold")

# ----------------------------
# Main App Body
# ----------------------------
st.title("🚦 Retention Ops Dashboard")
st.write("Real-time(ish) detection of high-risk users, action recommendations, and ROI-based prioritization.")

if data_file is None:
    st.info("Upload a feature snapshot CSV to begin.")
    st.stop()

# Load CSV
try:
    df = pd.read_csv(data_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Ensure id and needed features
id_col = pick_id_column(df)
df = ensure_no_purchase_30d(df)

# Choose feature set available in uploaded data
available_features = [c for c in REQUIRED_FEATURES if c in df.columns]
missing_features = [c for c in REQUIRED_FEATURES if c not in df.columns]

with st.expander("Detected schema"):
    st.write("**ID column:**", id_col)
    st.write("**Available features:**", available_features)
    if missing_features:
        st.warning(f"Missing recommended features: {missing_features}")

# Score customers
if model_file is not None:
    try:
        model = pickle.load(model_file)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    if not available_features:
        st.error("No overlapping features between CSV and REQUIRED_FEATURES. Either upload a pre-scored CSV or include more features.")
        st.stop()

    X = df[available_features].copy()
    # Fill numeric NaNs with median
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
    try:
        churn_score = score_with_model(model, X)
    except Exception as e:
        st.error(str(e))
        st.stop()

    df["churn_score"] = churn_score
else:
    if "churn_score" not in df.columns:
        st.error("No model uploaded and no `churn_score` column found. Upload a model or provide pre-scored data.")
        st.stop()

# Action recommendation & ROI
if "action" not in df.columns:
    df["action"] = df.apply(recommend_action, axis=1)

roi_parts = df.apply(lambda r: compute_roi_row(r, ltv_loss, success_map, cost_map), axis=1, result_type="expand")
for k in ["p_churn","p_success","expected_saved","cost","roi"]:
    df[k] = roi_parts[k]

# Treat decision
conditions = (
    (df["roi"] > 0) &
    (df["churn_score"] >= risk_threshold) &
    (df["p_success"] >= min_success_threshold)
)
df["treat"] = conditions.astype(int)

# Summary KPIs
total_users = len(df)
high_risk = int((df["churn_score"] >= risk_threshold).sum())
num_treat = int(df["treat"].sum())
total_expected_saved = float(df.loc[df["treat"]==1, "expected_saved"].sum())
total_cost = float(df.loc[df["treat"]==1, "cost"].sum())
net_uplift = total_expected_saved - total_cost

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Users in file", f"{total_users:,}")
col2.metric("High-risk (≥ thresh)", f"{high_risk:,}")
col3.metric("Recommended to treat", f"{num_treat:,}")
col4.metric("Expected saved (treated)", f"${total_expected_saved:,.0f}")
col5.metric("Net uplift (treated)", f"${net_uplift:,.0f}")

st.divider()

# Risk distribution
st.subheader("Risk Distribution")
st.caption("Churn score histogram. Use thresholds in sidebar to tune target size.")
st.bar_chart(df["churn_score"], use_container_width=True)

st.subheader("Priority Queue (Top N by ROI)")
N = st.slider("Show top N", min_value=10, max_value=2000, value=200, step=10)
cols_to_show = [id_col, "churn_score", "action", "p_success", "expected_saved", "cost", "roi", "treat",
                "avg_review_score","complaint_kw_any","low_star_flag","long_delivery_flag","low_freq_flag","no_purchase_30d","monetary"]
cols_to_show = [c for c in cols_to_show if c in df.columns]

priority_df = df.sort_values("roi", ascending=False).head(N)[cols_to_show]
st.dataframe(priority_df, use_container_width=True, hide_index=True)

# Export
csv = priority_df.to_csv(index=False).encode("utf-8")
st.download_button("Download target list (CSV)", data=csv, file_name="retention_targets.csv", mime="text/csv")

st.divider()

st.subheader("Action Mix & Economics")
action_mix = (
    df.loc[df["treat"]==1]
      .groupby("action")
      .agg(users=(id_col, "count"),
           avg_churn_score=("churn_score", "mean"),
           avg_p_success=("p_success", "mean"),
           total_expected_saved=("expected_saved", "sum"),
           total_cost=("cost", "sum"))
)
action_mix["net_uplift"] = action_mix["total_expected_saved"] - action_mix["total_cost"]

if len(action_mix):
    st.dataframe(action_mix.reset_index(), use_container_width=True)
else:
    st.info("No treated users under current thresholds.")

st.caption("Tip: Use this table to tune success rates/costs and see which actions drive the most net uplift.")

st.divider()

st.subheader("Ops Handoff Notes")
st.markdown("""
- **Coupon 10%**: Auto-send via CRM; exclude orders with ARPU < threshold if needed.
- **CS Callback + Apology Coupon**: Route to support with top complaint snippets (if available).
- **Shipping Voucher**: Generate one-time free shipping code; flag accounts with recent delays.
- **VIP Perk**: Add to loyalty tier; send concierge email.

> Treat rule: ROI > 0 AND churn_score ≥ threshold AND success_rate ≥ min threshold
""")
