
# 🔄 Data Science Workflow for: Churn & Retention Analysis in E-Commerce

---

## 1. Business Understanding
- Objective: Predict customer churn and build retention strategies
- Task Type: Prediction, classification, and behavior visualization
- Business Value: Increase CLTV, improve marketing efficiency, enhance user experience

---

## 2. Data Collection
- Source: Olist Brazilian E-Commerce dataset (Kaggle)
- Format: CSV files
- Tables Used:
  - Orders (`orders.csv`)
  - Reviews (`order_reviews.csv`)
  - Customer information (`customers.csv`)
  - Delivery details (`deliveries.csv`)
  - Product information (`products.csv`)

---

## 3. Data Understanding
- Initial inspection with `.head()`, `.info()`, `.describe()`, `.shape`
- Explore variable types, missing values, and data distributions
- Analyze key variables like review scores, delivery delays, purchase frequency
- Include EDA with visualizations and statistical summaries

---

## 4. Data Cleaning
- Remove or impute missing values (e.g., review text or delivery dates)
- Remove outliers (e.g., delivery delays over 200 days)
- Normalize date and categorical formats
- Drop duplicate records

---

## 5. Feature Engineering
- Create features such as:
  - `avg_review_score`: average rating per customer
  - `sentiment_score`: average sentiment from review texts
  - `avg_delivery_delay`: average number of delayed days
  - `purchase_count`: total purchases
  - `recency_days`: days since last purchase
  - `total_spent`: quantity × price (if applicable)

---

## 6. Train/Test Split
- Split customer data 80:20 using stratified sampling
- Use time-based split if dealing with sequential data
- Goal: avoid overfitting and ensure generalization

---

## 7. Modeling
- Baseline model: Logistic Regression
- Main models: Random Forest, XGBoost
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

---

## 8. Evaluation
- Metrics: Accuracy, AUC, Precision@TopK, Recall
- Visualization: Confusion Matrix, ROC Curve
- Focus on minimizing false negatives among high-risk customers

---

## 9. Interpretation & Insights
- Feature importance analysis using built-in methods or SHAP
- Key churn factors identified (e.g., delayed deliveries, negative reviews)
- Strategic insights for targeting high-risk segments

---

## 10. Deployment & Monitoring
- Deploy results using a Streamlit dashboard:
  - Input: customer ID
  - Output: churn risk score, behavioral timeline, top features
- Prepare retraining script for updated datasets
- Document and version codebase via GitHub with README and modular scripts
