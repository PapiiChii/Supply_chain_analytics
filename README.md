# 🚚 Predictive Supply Chain Pipeline: From SQL Staging to Explainable ML

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![Status](https://img.shields.io/badge/status-active-brightgreen)

An end-to-end pipeline that bridges relational database staging with a class-imbalanced machine learning model, built to flag high-risk order cancellations at the point of ingestion.

---

## 📌 Executive Summary

**The Business Problem**
E-commerce operations are traditionally reactive — logistics issues and customer cancellations are managed *after* they occur. In high-volume environments, late-stage order cancellations create supply chain friction, dead inventory allocation, and wasted shipping overhead.

**The Solution**
This pipeline extracts data from a local PostgreSQL instance hosting 110k+ transactional records, engineers customer behavior profiles, and trains a class-imbalanced Random Forest to flag high-risk cancellations at ingestion time.

**The Business Impact**
Working against an extreme **~0.5% minority class imbalance** (542 cancellations out of 110,739 orders), the model isolates a high-risk group of 200 orders per 21,838 test instances and captures **25% of true cancellations** within that flagged group. Routed to a customer intervention queue, this lets operations proactively target a small, high-risk slice of orders rather than reacting after the fact — though as detailed in [Section 3](#-3-predictive-modeling--evaluation), precision on the flagged group is currently low (13%), so this is best framed as an early-stage triage signal rather than a production-ready detector. See [Limitations & Next Steps](#-4-limitations--next-steps) for where this goes next.

---

## 🗄️ 1. Data Architecture & SQL Staging Layer

The foundational data layer is hosted locally in a **PostgreSQL** instance (`Supply_chain_analytics`). The target cohort is assembled via multi-table relational joins across staging schemas, mirroring the Olist e-commerce dataset structure.

### Entity Relationship & Schema Strategy

| Table | Role |
|---|---|
| `olist_orders_dataset (o)` | Source of truth for `order_id` and the operational `order_status` target |
| `olist_order_items_dataset (i)` | Granular ledger with financial metrics (`price`, `freight_value`) |
| `olist_products_dataset (p)` | Dimensional catalog with structural product metrics (weight, dimensions) |

### Optimized Extraction Query
[`queries/extract_cohort.sql`](queries/extract_cohort.sql)

```sql
SELECT
    o.order_id,
    o.order_status,
    i.price,
    i.freight_value,
    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,
    p.product_category_name AS category
FROM olist_orders_dataset o
JOIN olist_order_items_dataset i ON o.order_id = i.order_id
JOIN olist_products_dataset p ON i.product_id = p.product_id
WHERE o.order_status IN ('delivered', 'canceled');
```

---

## 📐 2. Feature Engineering & Preprocessing (Python)

Raw transactional logs aren't directly consumable by an ML model. The Python layer ([`src/ml_pipeline.py`](src/ml_pipeline.py)) transforms these inputs into behavioral and physical indicators using `pandas` and `scikit-learn`:

1. **`freight_ratio`** — `freight_value / price`. Captures consumer friction: when shipping cost approaches or exceeds item value, it's a strong signal for buyer's remorse and cancellation risk.
2. **`product_volume_cm3`** — Collapses three correlated spatial dimensions (`length × width × height`) into a single volumetric feature, reducing multicollinearity while preserving bulk/logistics signal.
3. **Target Binarization** — Remaps `order_status` (`delivered → 0`, `canceled → 1`) for algorithmic compatibility.

---

## 📊 3. Predictive Modeling & Evaluation

### Resolving Class Imbalance

The target vector has an acute real-world imbalance: **110,197 delivered vs. 542 canceled (~0.5%)**. A naive model predicting "delivered" every time would hit 99.5% accuracy while catching zero cancellations — accuracy alone is a misleading metric here.

To address this, the pipeline uses:
- A **stratified 80/20 train-test split** to preserve class ratios across both sets
- A **Random Forest Classifier** with `class_weight='balanced'` to penalize misclassification of the minority class

### Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | **0.7282** |
| True positives (cancellations caught) | 26 |
| Orders flagged as high-risk | 200 / 21,838 |

```text
📋 Detailed Classification Report

               precision    recall  f1-score   support

Delivered (0)       1.00      0.99      0.99     21732
 Canceled (1)       0.13      0.25      0.17       106

     accuracy                           0.99     21838
    macro avg       0.56      0.62      0.58     21838
 weighted avg       0.99      0.99      0.99     21838
```

**How to read this:** ROC-AUC of 0.73 shows the model has real, non-random discriminative power. But at the current operating threshold, precision on the flagged (canceled) class is only 0.13 — meaning roughly 1 in 8 flagged orders is an actual cancellation. Recall of 0.25 means the model catches a quarter of true cancellations within that flagged group. This tradeoff is a deliberate design choice for an intervention-queue use case (cheap to check a flagged order, costly to miss a real cancellation), but it also means the current model is a starting point, not a finished detector.

---

### 🧠 4 Explainable AI (SHAP) & Asset Visualization

To remove the "black box" constraints of ensemble tree learning, SHAP (SHapley Additive exPlanations) was implemented alongside standard feature importance matrices to isolate both the weight and direction of predictive attributes.

#### 🏆 Risk Driver Rankings (Feature Importance)
Absolute shipping economics govern user cancellation risks on the platform. The overall feature weights indicate that shipping costs (`freight_value`) and our engineered value proportions (`freight_ratio`) dictate **nearly 38%** of the model's total classification logic.

![Feature Importance](image_42f8e1.png)

#### 🧐 The Logistic Paradox (SHAP Summary Matrix Analysis)
While traditional feature importance reveals *which* variables hold predictive weight, the SHAP beeswarm summary matrix exposes a fascinating marketplace anomaly regarding *how* those features influence the target output:

![SHAP Summary Matrix](image_42f8a0.png)

*   **The Paradox:** High relative shipping costs typically signal financial friction. However, the SHAP matrix reveals that extreme, heavy freight costs (the prominent red clusters) actually push the prediction output significantly to the *left* of 0.0, mathematically driving down the cancellation probability and reinforcing a successful delivery outcome.
*   **Operational Conclusion:** Within the Olist ecosystem, exceptionally large or high-freight orders frequently represent high-commitment B2B logistics or specialized consumer purchases (e.g., heavy machinery, large furniture) where buyers expect high shipping friction and rarely cancel. Conversely, standard consumer shipments with lower, baseline freight variables carry the highest volume of erratic buyer behavior and cancellation variance.

---

## 🛠 5. Limitations & Next Steps

- **Low precision** means the intervention queue will contain many false positives; downstream cost of reviewing a flagged order should be cheap relative to a missed cancellation.
- **Threshold tuning:** the 200-order flag count reflects the default decision threshold — adjusting it via `predict_proba` and a custom cutoff could trade recall for precision depending on operational tolerance.
- **Feature expansion:** customer-level features (order history, prior cancellation rate, payment method, delivery estimate vs. actual) could meaningfully improve signal beyond product/pricing features alone.
- **Alternative approaches:** SMOTE/undersampling, gradient-boosted trees (XGBoost/LightGBM with `scale_pos_weight`), or anomaly-detection framings are worth benchmarking against the current Random Forest baseline.

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the SQL extraction (requires local PostgreSQL instance)
psql -d Supply_chain_analytics -f queries/extract_cohort.sql

# Run the ML pipeline
python src/ml_pipeline.py
```

## 📂 Project Structure

```
.
├── queries/
│   └── extract_cohort.sql
├── src/
│   └── ml_pipeline.py
├── requirements.txt
└── README.md
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
