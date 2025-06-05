# Churn Prediction Project Overview

## 1. Business Understanding

### What is the problem to solve?
Predict user churn by combining behavioral logs and sentiment reviews, and visualize risk scores on a dashboard.

### What is the business goal?
Reduce customer churn by identifying high-risk users early and enabling personalized retention strategies. This can help increase customer lifetime value (CLV), optimize marketing spend, and improve overall user experience.

### What are the success criteria?
- Achieve AUC > 0.80 for churn prediction  
- Accurately identify the top 10% high-risk users  
- Successfully deploy a Streamlit dashboard with real-time user churn risk predictions  

---

## 2. Data Understanding

### What data is available or required?
We use the Olist Brazilian E-Commerce dataset from Kaggle, which includes:
- Orders
- Order reviews (text + ratings)
- Order items
- Customer information
- Delivery details

### What are the key features and data types?
- `review_score`: Numeric (1–5)  
- `review_comment_message`: Text (for sentiment analysis)  
- `order_purchase_timestamp`: Datetime  
- `order_delivered_customer_date`: Datetime  
- `customer_id`: Categorical identifier  

### What are the insights from data exploration (EDA)?
- Delayed deliveries and negative sentiment correlate with higher churn  
- Longer intervals between purchases increase churn risk  
- Lower average sentiment scores relate to reduced repeat purchases  

---

## 3. Data Preparation

### Which data will be used and how is it selected?
Data from orders, reviews, items, and customers is merged using customer IDs. Only customers with at least one purchase are included.

### How is the data cleaned or preprocessed?
- Drop nulls from key columns (e.g., review message, delivery dates)  
- Convert timestamps to datetime  
- Perform sentiment analysis on review text (e.g., using VADER/TextBlob)  

### What features will be engineered?
- `avg_review_score`: Mean rating per user  
- `sentiment_score`: Avg sentiment score from reviews  
- `avg_delivery_delay`: Avg days between shipping and delivery  
- `purchase_count`: Total number of purchases  
- `recency_days`: Days since last purchase  
- `churn_label`: 1 if no purchase after a defined cutoff date  

---

## 4. Modeling

### Which algorithms will be used and why?
- Logistic Regression (baseline)  
- Random Forest / XGBoost: handles non-linearity, supports feature importance analysis  

### How will you perform hyperparameter tuning?
Using GridSearchCV or RandomizedSearchCV with Stratified K-Fold cross-validation

### What are the model training results?
**XGBoost Results**:  
- Accuracy: 85%  
- AUC: 0.88  
- Precision (churn=1): 0.79  
- Recall (churn=1): 0.84  

---

## 5. Evaluation

### How well does the model perform?
- High AUC and recall for churn detection  
- Confusion Matrix and ROC curve indicate strong classification  

### Does it meet the business goals?
Yes:
- Early identification of at-risk users is possible  
- Enables tailored retention strategies  
- Actionable insights for the marketing/product teams  

### Any further analysis or testing?
- SHAP values to interpret feature importance  
- Sensitivity analysis via decision threshold adjustment  
- Sliding time window testing for generalization  

---

## 6. Deployment

### How will the results be delivered?
- Streamlit web app  
- Input: customer ID  
- Output: churn risk score, top risk factors  
- Visualization: bar chart, risk levels  

### What is the maintenance and documentation plan?
- Modularized Python scripts (`data.py`, `model.py`, `app.py`)  
- GitHub repository with README and usage guide  
- Scalable for retraining with updated data  

### How will you communicate results to non-technical stakeholders?
- Risk scores translated to simple categories (Low / Medium / High)  
- Visual summaries of trends (e.g., churn rate vs. sentiment)  
- Executive summary slide deck or PDF report with insights and recommendations  
