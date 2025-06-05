## Workflow: Churn Prediction

### 1. Problem Definition
- Define churn as users who haven’t purchased in the past 90 days

### 2. Data Collection
- Files: `olist_orders.csv`, `olist_order_reviews.csv`, `olist_customers.csv`
- Join and clean datasets to generate a unified user timeline

### 3. Feature Engineering
- Purchase frequency, time since last order
- Review score averages, delivery delays
- Sentiment score from review messages

### 4. Modeling
- Split dataset into train/test
- Train models (XGBoost, RandomForest)
- Tune hyperparameters

### 5. Evaluation
- Use ROC-AUC, Precision, Recall
- Identify high-risk users

### 6. Visualization / Deployment
- Streamlit dashboard to present churn scores
- Optional: Retention suggestion based on churn probability
