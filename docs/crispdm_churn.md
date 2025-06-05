## CRISP-DM: Churn Prediction Project

### 1. Business Understanding
- **Goal**: Predict user churn (i.e., users who stop purchasing for 90+ days).
- **Business Value**: Retain loyal users, increase customer lifetime value (CLV), reduce marketing spend.
- **Success Criteria**: 
  - Achieve AUC > 0.80
  - Accurately identify the top 10% high-risk users

### 2. Data Understanding
- **Data Source**: Olist dataset (orders, reviews, customers, items, products)
- **Key Features**:
  - Purchase frequency
  - Delivery delay
  - Review sentiment score

### 3. Data Preparation
- Handle missing values and outliers
- Convert dates, normalize features
- Generate user-level feature table

### 4. Modeling
- **Models**: XGBoost, RandomForest, Logistic Regression
- **Target**: Users with no purchases in 90+ days

### 5. Evaluation
- Metrics: ROC-AUC, Confusion Matrix
- Compare models and select best performing

### 6. Deployment
- Build a Streamlit dashboard
- Display real-time churn risk scores
