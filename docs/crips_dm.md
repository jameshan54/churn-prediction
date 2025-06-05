
# 🌀 CRISP-DM for: Churn & Retention Analysis in E-Commerce

---

## 1. Business Understanding

**What is the problem to solve?**  
Predict customer churn early, visualize behavioral patterns of high-risk users, and design data-driven retention strategies.

**What is the business goal?**  
- Increase Customer Lifetime Value (CLTV)  
- Optimize marketing spend  
- Improve overall user experience

**What are the success criteria?**  
- AUC ≥ 0.80  
- Accurately identify the top 10% of high-risk churn users  
- Deliver visual behavior mapping and retention simulation scenarios

---

## 2. Data Understanding

**Data Source:**  
- Olist Brazilian E-commerce dataset (Kaggle)  
- Total of 10 relational tables

**Key Tables:**  
- `orders`, `order_items`, `order_reviews`, `customers`, `products`, `deliveries`

**What are the key features and data types?**
- `review_score`: Numeric (1–5)  
- `review_comment_message`: Text (for sentiment analysis)  
- `order_purchase_timestamp`: Datetime  
- `order_delivered_customer_date`: Datetime  
- `customer_id`: Categorical identifier

**Insights from Data Exploration (EDA):**
- Delayed deliveries and negative sentiment correlate with higher churn  
- Longer intervals between purchases increase churn risk  
- Lower average sentiment scores reduce likelihood of repeat purchases

---

## 3. Data Preparation

**Which data will be used and how is it selected?**  
- Merged data from `orders`, `reviews`, `items`, and `customers` using `customer_id`  
- Only customers with at least one completed purchase are included

**How is the data cleaned or preprocessed?**  
- Dropped nulls from essential fields (e.g., review message, delivery dates)  
- Converted timestamps to datetime format  
- Performed sentiment analysis on review texts using VADER/TextBlob

**What features are engineered?**  
- `avg_review_score`: Average rating per customer  
- `sentiment_score`: Mean sentiment score from reviews  
- `avg_delivery_delay`: Mean delivery delay in days  
- `purchase_count`: Number of purchases  
- `recency_days`: Days since last purchase  
- `churn_label`: Binary label (1 = churn, based on inactivity after a cutoff date)

---

## 4. Modeling

**Which algorithms are used and why?**  
- Logistic Regression (as a baseline)  
- Random Forest / XGBoost (handle non-linear relationships, interpretable, robust to outliers)

**How is hyperparameter tuning performed?**  
- GridSearchCV or RandomizedSearchCV  
- Stratified K-Fold cross-validation

**What are the model results?**  
Using XGBoost:
- Accuracy: 85%  
- AUC: 0.88  
- Precision (churn=1): 0.79  
- Recall (churn=1): 0.84

---

## 5. Evaluation

**How well does the model perform?**  
- High AUC and recall for detecting churned users  
- ROC curve and Confusion Matrix confirm strong classification capability

**Does it meet the business goals?**  
✅ Enables early identification of at-risk users  
✅ Provides actionable insights for marketing and product teams  
✅ Supports data-driven retention planning

**Further analysis or testing?**  
- SHAP values for feature interpretation  
- Threshold tuning for cost-sensitive use cases  
- Sliding time window validation for robustness

---

## 6. Deployment

**How will the results be delivered?**  
- **Streamlit Web App**  
  - **Input:** Customer ID  
  - **Output:** Churn risk score + top contributing factors  
  - **Visualization:** Bar charts, risk levels, user behavior timeline

**What is the maintenance & documentation plan?**  
- Modularized codebase (`data.py`, `model.py`, `app.py`)  
- GitHub repo with README, instructions, and retraining scripts  
- Easily updatable for future datasets

**How will results be communicated to non-technical stakeholders?**  
- Risk levels categorized into "Low", "Medium", "High"  
- Visual summaries of churn vs. sentiment trends  
- Executive summary slide deck with insights and retention strategy recommendations
