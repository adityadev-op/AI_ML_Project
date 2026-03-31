# Project Report: High-value Customers Identification for an E-Commerce Company

## 1. Problem Statement
Identify high-value customers who contribute most to revenue and lifetime value (CLV) for an e-commerce business. This helps prioritize retention, promotion, and personalized offers.

## 2. Why it Matters
- In e-commerce, 20% of customers often generate 80% of revenue (Pareto Principle).
- Identifying these customers reduces churn and increases ROI for marketing budgets.
- Enables segmentation for loyalty programs, discounts, and premium services.

## 3. Approach
1. Generate a synthetic dataset of 10,000 customers with features: orders, revenue, avg order value, recency, returns, tenure, category preferences.
2. Create a target label `high_value` as top 20% customers by total revenue.
3. Apply preprocessing + scaling.
4. Train classification models (Logistic Regression, RandomForest).
5. Evaluate using accuracy, ROC-AUC, precision, recall, F1 score.

## 4. Key Decisions
- Synthetic data due to absence of provided dataset.
- Top-20% revenue as high-value definition; aligns with business insights.
- Balanced classification with metrics beyond accuracy (precision/recall) because of class imbalance.
- RandomForest chosen for interpretability of feature importance and robust performance.

## 5. Challenges
- No real data; needed credible synthetic generation.
- Avoiding data leakage by splitting before scaling.
- Ensuring reproducible results with fixed random seed.

## 6. Results
- Model metrics are printed by `src/run.py` (e.g., accuracy > 0.85, ROC-AUC > 0.90 for benchmark run).
- Feature importances show `total_revenue`, `avg_order_value`, `frequency` as top contributors.

## 7. Learning
- End-to-end pipeline structure is critical for maintainability.
- Feature engineering (RFM-based metrics) drives performance more than model complexity.
- Documentation and submission artifacts are as important as code.

## 8. Future Work
- Replace synthetic generator with real transaction logs and customer profile data.
- Add time-series CLV and survival modeling.
- Build interactive dashboard for business users.
