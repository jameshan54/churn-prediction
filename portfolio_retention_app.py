import numpy as np
import pandas as pd
import pickle
import streamlit as st
from typing import Dict, List

# ============================
# Page / Theme
# ============================
st.set_page_config(
    page_title="Retention AI Dashboard – Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Sidebar background */
[data-testid="stSidebar"] { background: #f6f8fb; }

/* Buttons */
div.stButton > button {
  color: white; background: #00ADB5; border-radius: 10px; height: 3em; font-weight: 600;
}

/* Metrics */
.css-12w0qpk, .css-1xarl3l { color: #0f172a; }

/* Headings */
h1, h2, h3 { color: #00ADB5; }

/* DataFrame body */
.stDataFrame tbody td { font-size: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================
# Constants
# ============================
REQUIRED_FEATURES: List[str] = [
    "total_orders","avg_order_value","avg_num_items","avg_unique_products",
    "avg_actual_delivery_days","tenure_days","order_frequency","monetary",
    "avg_review_score","last_review_score","review_count",
    "complaint_kw_any","complaint_kw_ratio",
    "low_freq_flag","long_delivery_flag","low_star_flag","no_purchase_30d"
]
ID_CANDIDATES = ["customer_unique_id", "customer_id", "id", "user_id"]

# ============================
# Utility
# ============================
def pick_id_column(df: pd.DataFrame) -> str:
    for c in ID_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[0]


def ensure_no_purchase_30d(df: pd.DataFrame) -> pd.DataFrame:
    if "no_purchase_30d" not in df.columns:
        if "recency_days" in df.columns:
            df["no_purchase_30d"] = (df["recency_days"] > 30).astype(int)
        else:
            df["no_purchase_30d"] = 0
    return df


def compute_ltv(arpu_per_order: float, months: float, orders_per_month: float) -> float:
    return max(arpu_per_order, 0) * max(months, 0) * max(orders_per_month, 0)


def score_with_model(model_obj, X: pd.DataFrame) -> np.ndarray:
    try:
        return model_obj.predict_proba(X)[:, 1]
    except Exception as e:
        raise RuntimeError(f"Model scoring failed: {e}")


def make_demo_dataframe(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_unique_id": [f"demo_{i:05d}" for i in range(n)],
        "total_orders": rng.integers(1, 8, n),
        "avg_order_value": rng.normal(120, 40, n).clip(5),
        "avg_num_items": rng.normal(2.2, 0.8, n).clip(1),
        "avg_unique_products": rng.normal(1.6, 0.6, n).clip(1),
        "avg_actual_delivery_days": rng.normal(7, 3, n).clip(0),
        "tenure_days": rng.integers(0, 900, n),
        "order_frequency": rng.normal(1.0, 0.5, n).clip(0.05),
        "monetary": rng.normal(450, 220, n).clip(10),
        "avg_review_score": rng.normal(4.0, 0.6, n).clip(1, 5),
        "last_review_score": rng.integers(1, 6, n),
        "review_count": rng.integers(0, 20, n),
        "complaint_kw_any": rng.choice([0,1], n, p=[0.85, 0.15]),
        "complaint_kw_ratio": rng.beta(1, 10, n),
        "low_freq_flag": 0,  # 잠시 0으로 두고 아래서 median기반으로 다시 계산
        "long_delivery_flag": 0,
        "low_star_flag": 0,
        "no_purchase_30d": rng.choice([0,1], n, p=[0.6, 0.4]),
    })
    # 파생 플래그를 데이터에 맞춰 조정
    df["low_freq_flag"] = (df["order_frequency"] < df["order_frequency"].median()).astype(int)
    df["long_delivery_flag"] = (df["avg_actual_delivery_days"] > df["avg_actual_delivery_days"].median()).astype(int)
    df["low_star_flag"] = (df["avg_review_score"] < 3.5).astype(int)

    # 합성 churn_score (시연용): 로지스틱 결합
    z = (
        0.8*(df["low_freq_flag"]) +
        0.7*(df["no_purchase_30d"]) +
        0.6*(df["long_delivery_flag"]) +
        0.5*(df["low_star_flag"]) +
        0.4*(df["complaint_kw_any"]) -
        0.002*(df["monetary"]) -
        0.1*(df["order_frequency"]) +
        0.0005*(df["tenure_days"]) -
        0.01*(df["avg_review_score"]) +
        rng.normal(0, 0.3, len(df))
    )
    df["churn_score"] = 1/(1+np.exp(-z))
    return df

# ============================
# Sidebar (Simple vs Advanced)
# ============================
with st.sidebar:
    st.image(
        "https://static.streamlit.io/examples/dice.jpg",
        caption="Retention AI",
        width=160,
    )
    st.header("⚙️ Settings")
    simple_mode = st.toggle("Simple mode (recommended)", value=True, help="Hide expert knobs. Use presets and auto-filled inputs.")

    st.subheader("Data & Model")
    use_demo = st.checkbox("Use demo data", value=True, help="Try the app without uploading anything.")
    data_file = None
    model_file = None
    if not use_demo:
        data_file = st.file_uploader("Upload feature snapshot CSV", type=["csv"]) 
        model_file = st.file_uploader("Upload trained model (.pkl)", type=["pkl"]) 
        st.caption("CSV should include columns similar to REQUIRED_FEATURES. If no model is provided, CSV must have `churn_score`.")

# ============================
# Load Data
# ============================
if use_demo:
    df = make_demo_dataframe(n=1200)
else:
    if data_file is None:
        st.info("Upload a CSV or switch on demo mode to continue.")
        st.stop()
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

# Ensure basics
id_col = pick_id_column(df)
df = ensure_no_purchase_30d(df)

# ============================
# Simple Presets & Auto-fill
# ============================
presets = {
    "Conservative": {"risk_p": 0.95, "min_sr": 0.40, "months": 2.0},
    "Balanced":     {"risk_p": 0.80, "min_sr": 0.30, "months": 3.0},
    "Aggressive":   {"risk_p": 0.60, "min_sr": 0.20, "months": 4.0},
}

if simple_mode:
    st.sidebar.subheader("🎯 Targeting Preset")
    preset_name = st.sidebar.selectbox("Select preset", list(presets.keys()), index=1)
    pconf = presets[preset_name]

    # Risk threshold by percentile
    if "churn_score" in df.columns:
        risk_threshold = float(np.quantile(df["churn_score"].values, pconf["risk_p"]))
    else:
        risk_threshold = 0.5  # fallback

    min_success_threshold = pconf["min_sr"]

    # Auto-fill LTV inputs from data
    with np.errstate(divide='ignore', invalid='ignore'):
        arpu_auto = (df.get("monetary", pd.Series([100])).astype(float) / df.get("total_orders", pd.Series([1])).astype(float))
    arpu_auto = float(np.nanmedian(arpu_auto.replace([np.inf,-np.inf], np.nan))) if np.isfinite(arpu_auto).any() else 100.0
    orders_per_month_auto = float(np.nanmedian(df.get("order_frequency", pd.Series([1.0])).astype(float)))
    retention_months = pconf["months"]

    st.sidebar.subheader("💰 LTV (auto)")
    arpu = st.sidebar.number_input("ARPU per order", min_value=0.0, value=float(round(arpu_auto,2)), step=1.0)
    orders_per_month = st.sidebar.number_input("Expected orders per month", min_value=0.0, value=float(round(orders_per_month_auto,2)), step=0.1)
    st.sidebar.caption("Retention window is preset by mode; switch to Advanced to change.")
else:
    st.sidebar.subheader("Thresholds")
    risk_threshold = st.sidebar.slider("Churn score threshold", 0.0, 1.0, 0.50, 0.01)
    min_success_threshold = st.sidebar.slider("Min retention success rate threshold", 0.0, 1.0, 0.30, 0.01)

    st.sidebar.subheader("LTV Inputs")
    arpu = st.sidebar.number_input("ARPU per order", min_value=0.0, value=100.0, step=1.0)
    retention_months = st.sidebar.number_input("Retention window (months)", min_value=0.0, value=3.0, step=0.5)
    orders_per_month = st.sidebar.number_input("Expected orders per month", min_value=0.0, value=1.0, step=0.1)

ltv_loss = compute_ltv(arpu, retention_months, orders_per_month)
st.sidebar.metric("Estimated LTV loss if churned (per user)", f"${ltv_loss:,.2f}")

# Success rates & costs
if simple_mode:
    success_map: Dict[str, float] = {
        "Coupon 10%": 0.30,
        "CS Callback + Apology Coupon": 0.35,
        "Shipping Voucher": 0.32,
        "VIP Perk": 0.28,
    }
    cost_map: Dict[str, float] = {
        "Coupon 10%": 8.0,
        "CS Callback + Apology Coupon": 10.0,
        "Shipping Voucher": 7.0,
        "VIP Perk": 12.0,
    }
else:
    st.sidebar.subheader("Action Success Rates (editable)")
    sr_coupon = st.sidebar.slider("Coupon 10%", 0.0, 1.0, 0.30, 0.01)
    sr_cs = st.sidebar.slider("CS Callback + Apology Coupon", 0.0, 1.0, 0.35, 0.01)
    sr_ship = st.sidebar.slider("Shipping Voucher", 0.0, 1.0, 0.32, 0.01)
    sr_vip = st.sidebar.slider("VIP Perk", 0.0, 1.0, 0.28, 0.01)
    success_map = {"Coupon 10%": sr_coupon, "CS Callback + Apology Coupon": sr_cs, "Shipping Voucher": sr_ship, "VIP Perk": sr_vip}

    st.sidebar.subheader("Action Costs (editable)")
    cost_coupon = st.sidebar.number_input("Coupon 10% cost (avg)", min_value=0.0, value=8.0, step=1.0)
    cost_cs = st.sidebar.number_input("CS Callback + Apology Coupon cost", min_value=0.0, value=10.0, step=1.0)
    cost_ship = st.sidebar.number_input("Shipping Voucher cost", min_value=0.0, value=7.0, step=1.0)
    cost_vip = st.sidebar.number_input("VIP Perk cost", min_value=0.0, value=12.0, step=1.0)
    cost_map = {"Coupon 10%": cost_coupon, "CS Callback + Apology Coupon": cost_cs, "Shipping Voucher": cost_ship, "VIP Perk": cost_vip}

st.sidebar.caption("Treat if: ROI > 0 AND churn_score ≥ threshold AND success_rate ≥ min threshold")

# ============================
# Scoring (if model provided)
# ============================
available_features = [c for c in REQUIRED_FEATURES if c in df.columns]
missing_features = [c for c in REQUIRED_FEATURES if c not in df.columns]

if not use_demo and model_file is not None:
    try:
        model = pickle.load(model_file)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    if not available_features:
        st.error("No overlapping features between CSV and REQUIRED_FEATURES. Either upload a pre-scored CSV or include more features.")
        st.stop()

    X = df[available_features].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
    try:
        df["churn_score"] = score_with_model(model, X)
    except Exception as e:
        st.error(str(e))
        st.stop()

# If still no churn_score, create a light-weight heuristic for demo
if "churn_score" not in df.columns:
    # fallback: normalize a heuristic combination
    h = (
        0.4*df.get("no_purchase_30d", pd.Series(0)) +
        0.3*df.get("low_freq_flag", pd.Series(0)) +
        0.2*df.get("long_delivery_flag", pd.Series(0)) +
        0.1*df.get("low_star_flag", pd.Series(0))
    ).astype(float)
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    df["churn_score"] = h

# ============================
# Action recommendation & ROI
# ============================

def recommend_action(row: pd.Series) -> str:
    if int(row.get("complaint_kw_any", 0)) == 1 or int(row.get("low_star_flag", 0)) == 1:
        return "CS Callback + Apology Coupon"
    if int(row.get("long_delivery_flag", 0)) == 1:
        return "Shipping Voucher"
    if float(row.get("monetary", 0)) >= float(df.get("monetary", pd.Series([0])).median()):
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
    return {"p_churn": p_churn, "p_success": p_success, "expected_saved": expected_saved, "cost": cost, "roi": roi}

if "action" not in df.columns:
    df["action"] = df.apply(recommend_action, axis=1)

roi_parts = df.apply(lambda r: compute_roi_row(r, ltv_loss, success_map, cost_map), axis=1, result_type="expand")
for k in ["p_churn","p_success","expected_saved","cost","roi"]:
    df[k] = roi_parts[k]

# Treat decision
conditions = (
    (df["roi"] > 0) & (df["churn_score"] >= risk_threshold) & (df["p_success"] >= min_success_threshold)
)
df["treat"] = conditions.astype(int)

# ============================
# KPIs
# ============================
total_users = len(df)
high_risk = int((df["churn_score"] >= risk_threshold).sum())
num_treat = int(df["treat"].sum())
total_expected_saved = float(df.loc[df["treat"]==1, "expected_saved"].sum())
total_cost = float(df.loc[df["treat"]==1, "cost"].sum())
net_uplift = total_expected_saved - total_cost

# ============================
# Header
# ============================
colH1, colH2 = st.columns([3,1])
with colH1:
    st.markdown("## 🧠 Retention AI Dashboard – Portfolio")
    st.caption("Churn risk • Action recommendations • ROI optimization")
with colH2:
    st.metric("Net uplift", f"${net_uplift:,.0f}")

# ============================
# Tabs
# ============================
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk", "Actions", "Data"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{total_users:,}")
    c2.metric("High-risk (≥ thresh)", f"{high_risk:,}")
    c3.metric("Recommended to treat", f"{num_treat:,}")
    c4.metric("Expected saved", f"${total_expected_saved:,.0f}")

    st.divider()
    st.subheader("Priority Queue (Top N by ROI)")
    N = st.slider("Show top N", min_value=10, max_value=2000, value=200, step=10)
    cols_to_show = [id_col, "churn_score", "action", "p_success", "expected_saved", "cost", "roi", "treat",
                    "avg_review_score","complaint_kw_any","low_star_flag","long_delivery_flag","low_freq_flag","no_purchase_30d","monetary"]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    priority_df = df.sort_values("roi", ascending=False).head(N)[cols_to_show]
    st.dataframe(priority_df, use_container_width=True, hide_index=True)

    csv = priority_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download target list (CSV)", data=csv, file_name="retention_targets.csv", mime="text/csv")

with tab2:
    st.subheader("Churn score distribution")
    bins = np.linspace(0, 1, 31)
    hist, edges = np.histogram(df["churn_score"], bins=bins)
    centers = 0.5*(edges[1:]+edges[:-1])
    hist_df = pd.DataFrame({"count": hist}, index=np.round(centers, 2))
    st.bar_chart(hist_df, use_container_width=True)

    st.caption("Tip: In Simple mode, the threshold follows a percentile preset (Conservative/Balanced/Aggressive). In Advanced mode, you can directly set the threshold.")

with tab3:
    st.subheader("Action Mix & Economics (treated only)")
    action_mix = (
        df.loc[df["treat"]==1]
          .groupby("action")
          .agg(users=(id_col, "count"),
               avg_churn_score=("churn_score", "mean"),
               avg_p_success=("p_success", "mean"),
               total_expected_saved=("expected_saved", "sum"),
               total_cost=("cost", "sum"))
    )
    if len(action_mix):
        action_mix["net_uplift"] = action_mix["total_expected_saved"] - action_mix["total_cost"]
        st.dataframe(action_mix.reset_index(), use_container_width=True)
    else:
        st.info("No treated users under current thresholds.")

    st.markdown("""
**Ops Handoff Notes**
- **Coupon 10%**: Auto-send via CRM; exclude low-ARPU orders if needed.
- **CS Callback + Apology Coupon**: Route to support with top complaint snippets (if available).
- **Shipping Voucher**: Generate one-time shipping code; flag accounts with recent delays.
- **VIP Perk**: Add to loyalty tier; concierge email.

> Treat if: ROI > 0 AND churn_score ≥ threshold AND success_rate ≥ min threshold
""")

with tab4:
    with st.expander("Detected schema", expanded=False):
        st.write("**ID column:**", id_col)
        st.write("**Available features:**", available_features)
        if missing_features:
            st.warning(f"Missing recommended features: {missing_features}")
    st.dataframe(df.head(50), use_container_width=True)

st.caption("Made with ❤️ for portfolio demos. Simple mode keeps knobs minimal; switch to Advanced for full control.")
