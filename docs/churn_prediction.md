# Churn Prediction

## 🧠 Objective
Predict high-risk users likely to churn based on behavioral and delivery data.

## 📦 Data Used
- `orders.csv`, `customers.csv`, `order_items.csv`
- Key columns: `order_purchase_timestamp`, `order_delivered_customer_date`, `customer_id`
- Derived features: `days_to_deliver`, `purchase_frequency`, `recency`, `product_variety`

## 🔧 Preprocessing
- Handled missing values and inconsistent timestamps
- Calculated delivery delay, recency, frequency, and user-level aggregates
- Normalized numeric features for modeling

## 🧪 Modeling
- Algorithms: **XGBoost**, Logistic Regression
- Split: Time-based train/test (chronological)
- Metrics: AUC, F1-score, Precision-Recall

## 📊 Results
- Best model: **XGBoost**
- AUC: **0.83**
- Key features:
  - Delivery delay
  - Inactivity period
  - Order frequency
  - Number of distinct product categories

## 💡 Insights
- Users with longer delivery delays and long inactivity gaps were more likely to churn.
