# Sentiment & Behavioral Feature Analysis

## 🧠 Objective
Identify churn drivers by combining review sentiment with behavioral signals.

## 📦 Data Used
- `order_reviews.csv`:
  - `review_score`, `review_comment_message`
- `orders.csv`, `order_items.csv` for behavioral context

## 🔍 Sentiment Analysis
- Preprocessing: lowercasing, removing punctuation
- Tool: **VADER** (Valence Aware Dictionary for Sentiment Reasoning)
- Feature created: `review_sentiment_score` per order

## 📐 Feature Engineering
- Combined:
  - `delivery_delay × negative_sentiment`
  - `review_score trend over time`
  - `negative_sentiment frequency`
- Used correlation matrix to identify key signals

## 📊 Results
- Strong churn correlation with:
  - Low review score AND late delivery
  - Negative sentiment in consecutive orders

## 💡 Insights
- Users express early warning signals through negative feedback.
- Behavioral anomalies amplify the emotional dissatisfaction leading to churn.
