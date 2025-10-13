# E-Commerce Customer Churn Prediction

The goal of this End-to-End project is to:

1. Predict **customer churn risk** on an e-commerce platform.  
2. Combine **sentiment analysis** with **behavioral data** to proactively identify high-risk customers at an early stage.  
3. Enable the development of personalized **retention strategies** to maximize customer lifetime value.  

---

# (practice) Tableau Visualization for class
<p align="center">
  <img src="https://github.com/user-attachments/assets/91c68343-212d-4248-9758-751ef0ba2b01" width="500" />
</p>

# Progress so far (10/12 Sunday)

**1. Ran the initial classification model**
- Built a churn prediction model using a 90-day definition.
- The dataset was extremely imbalanced - churn rate around **90%**, repurchase rate only **3%**.
- The model predicted almost all users as **churn = 1**, and performance for **churn = 0** (retained users) was very poor.

**2. Tried to handle class imbalance**
- Applied **class weighting, SMOTE (oversampling), SMOTETomek**, and **threshold tuning**.
- Even considering trade-offs between recall and precision, the model still predicted most users as churn = 1, and performance for churn = 0 did not significantly improve.

**3. Adjusted the churn window**
- Changed the churn definition from **90 days -> 150 days.**
- The ratio improved to about **80% churn / 20% non-churn**, which made the data slightly more balanced.
- However, **AUC and accuracy both decreased**, likely because the broader churn definition introduced more noise and blurred behavioral differences.

 4. **Realization - classification accuracy is meaningless under extreme imbalance.**
- The imbalance was too severe for traditional classification metrics to be meaningful.
- Decided to **reframe the problem** from classification to a **ranking / probability model**,
focusing on estimating each customer’s churn probability rather than a binary prediction.

5. Future work
- Add **behavioral + sentiment features** (e.g., review text, complaint ratio)
to see if model performance improves — this part is partially done and will be merged later.
- Use **Feature Importance** and **SHAP** for model interpretation and key driver analysis.
