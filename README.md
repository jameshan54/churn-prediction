# End-to-End Customer Churn Prediction

This project consists of **three main parts**, representing a full data science lifecycle from modeling to deployment.

| Part | What happens here | File |
|-----:|--------------------|------|
| 1️⃣ | **Imbalance diagnosis → reframing from binary classification to ranking/probability.** Create churn labels & baseline sanity checks. | `01_churn_prediction.ipynb` |
| 2️⃣ | **Feature Integration (Sentiment & Behavior) → Probability/Ranking Modeling.** Merge RFM, delivery delays, review sentiment, product diversity, etc., then train calibrated ranking/probability models (AUC, Precision@K, Lift). | `02_ranking_model.ipynb` |
| 3️⃣ | **Deployment – Retention Strategy App.** Visualize churn risk deciles and simulate actions/costs. | `03_streamlit_retention_app.py` |

**Flow:** data → **sentiment/behavior features** → reframing → **feature integration + ranking model** → deployment
> The project reframes churn prediction from a highly imbalanced binary classification task into a ranking-based probability modeling pipeline, enabling actionable retention insights.


---

# Progress so far (10/12 Sunday)

**1. Ran the initial classification model**
- Built a churn prediction model using a 90-day definition.
- The dataset was extremely imbalanced - churn rate around **90%**, repurchase rate only **3%**.
- The model predicted almost all users as **churn = 1**, and performance for **churn = 0** (retained users) was very poor.

**1.1 Tried to handle class imbalance**
- Applied **class weighting, SMOTE (oversampling), SMOTETomek**, and **threshold tuning**.
- Even considering trade-offs between recall and precision, the model still predicted most users as churn = 1, and performance for churn = 0 did not significantly improve.

**1.2 Adjusted the churn window**
- Changed the churn definition from **90 days -> 150 days.**
- The ratio improved to about **80% churn / 20% non-churn**, which made the data slightly more balanced.
- However, **AUC and accuracy both decreased**, likely because the broader churn definition introduced more noise and blurred behavioral differences.

**2. Realization - classification accuracy is meaningless under extreme imbalance.**
- The imbalance was too severe for traditional classification metrics to be meaningful.
- Decided to **reframe the problem** from classification to a **ranking / probability model**,
focusing on estimating each customer’s churn probability rather than a binary prediction.

**3. Future work**
- Add **behavioral + sentiment features** (e.g., review text, complaint ratio)
to see if model performance improves — this part is partially done and will be merged later.
- Use **Feature Importance** and **SHAP** for model interpretation and key driver analysis.



