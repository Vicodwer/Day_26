# 🛵 ZeptoFresh – 15-Minute Food & Essentials Delivery
### EDA & Late Delivery Risk Prediction Case Study
> PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar — Week 05, Day 27

---

## 📌 Problem Statement

ZeptoFresh operates **350+ micro-fulfillment hubs across 25 Indian cities**, promising 15-minute delivery of groceries and ready-to-eat meals. The analytics team is performing **Exploratory Data Analysis (EDA)** on 110,000 order records from Q2 2025 before building a **late delivery risk prediction model**.

> **Late Delivery Threshold:** Deliveries taking 30+ minutes significantly increase the probability of customer churn.

---

## 📂 Dataset Overview

| Property | Details |
|---|---|
| Records | 110,000 orders |
| Period | Q2 2025 |
| Cities | 25 Indian cities |

### Dataset Columns

| Column | Type | Description |
|---|---|---|
| `order_id` | ID | Unique order identifier |
| `hub_id` | Categorical | Fulfillment hub |
| `city` | Categorical | Delivery city |
| `order_category` | Categorical | Grocery / Fresh Food / Bakery / Medicines |
| `order_value_Rs` | Numeric | Order value in ₹ |
| `items_count` | Numeric | Number of items |
| `delivery_time_mins` | Numeric | **Target-related** — actual delivery duration |
| `prep_time_mins` | Numeric | Kitchen preparation time |
| `rider_distance_km` | Numeric | Distance covered by rider |
| `order_hour` | Numeric | Hour of day order was placed |
| `is_weekend` | Binary | Weekend flag |
| `rain_flag` | Binary | Rainy weather indicator |
| `customer_age` | Numeric | Age of customer |
| `customer_tenure_days` | Numeric | Days since account creation |
| `coupon_used` | Binary | Discount coupon applied |
| `tip_amount_Rs` | Numeric | Tip given by customer |
| `refund_issued` | Binary | Whether refund was issued |
| `customer_rating` | Float (1–5) | Post-delivery rating |

---

## 🔍 EDA Observations Summary

| Observation | Column | Finding |
|---|---|---|
| 1 | `delivery_time_mins` | Min=0, Max=142, Mean=18.4, Median=14.2 — right-skewed; 214 rows = 0 |
| 2 | `order_value_Rs` | Max=₹2.95L (anomaly); Mean=₹620, Median=₹310 |
| 3 | `prep_time_mins` | Contains negative values; dtype = int64 |
| 4 | `customer_rating` | 9,800 nulls; values of 0 outside valid 1–5 scale |
| 5 | Correlation Matrix | delivery↔refund: r=0.74, rain↔delivery: r=0.48, tip↔rating: r=0.63 |
| 6 | Distribution Shape | Bimodal in Mumbai & Bangalore (peaks at 12–14 and 28–32 min) |

---

## 🩺 (a) Data Quality Diagnosis

### Problem 1 — `delivery_time_mins = 0` (214 rows)
- **Type:** Data Entry Error / Impossible Value
- **Treatment:** Remove or impute using median `delivery_time_mins` grouped by `hub_id` + `order_hour`

### Problem 2 — `order_value_Rs = ₹2,95,000`
- **Type:** Outlier (likely erroneous)
- **Treatment:** Winsorize at 99th percentile; cross-verify if it's a legitimate B2B order to be segmented separately

### Problem 3 — Negative values in `prep_time_mins`
- **Type:** Structural / Data Entry Error
- **Treatment:** Replace negatives with `NaN`; impute using median `prep_time_mins` grouped by `order_category`

### Problem 4 — `customer_rating`: 9,800 nulls + values of 0
- **Type:** Missing Values + Out-of-Range Values
- **Treatment:** Treat 0-ratings as missing; create binary flag `rating_given = 0/1`; retain nulls as informative missingness rather than imputing

---

## 📈 (b) Distribution Analysis

### Mean (18.4) > Median (14.2) → Right-Skewed Distribution

```
Frequency
  |
  |  ████
  |  ████ ██
  |  ████ ████
  |  ████ ████ ██
  |  ████ ████ ████ ██  █
  +--+----+----+----+----+----+---> delivery_time_mins
     8   14   20   28   38  60+

       ↑ Most orders     ↑ Long tail
       cluster here      (outlier delays)
```

### Transformation: `log(delivery_time_mins + 1)`

- Compresses the long right tail → approximately normal distribution
- `+1` offset prevents `log(0) = -∞` for zero-value rows
- Required for linear/logistic regression which assumes near-normal feature distributions

---

## 🔗 (c) Correlation Interpretation

### Claim: *"Late deliveries cause refunds. Solving delay will eliminate refunds."*

**What's logically incorrect:**
Correlation ≠ Causation. `r = +0.74` indicates a strong positive association, not a causal relationship. The direction of causality, or a third variable driving both, cannot be determined from correlation alone.

**What r = +0.74 actually means:**
When `delivery_time_mins` is high, `refund_issued` tends to be high — they co-vary. The relationship may be indirect or mediated.

**Possible Confounders:**

| Confounder | Explanation |
|---|---|
| `rain_flag` | Rain causes delays AND damaged packaging → refunds independently |
| `order_category` | Fresh Food / Medicines are time-sensitive; late AND more likely to be refunded |

> ⚠️ Fixing delays alone may not eliminate refunds if the true driver is a confounder like rain or product type.

---

## 🔔 (d) Bimodal Pattern in Tier-1 Cities (Mumbai & Bangalore)

### Peaks: 12–14 mins AND 28–32 mins

**Operational Reasons:**
1. **Two zone types** — dense urban cores (fast) vs. congested arterials / high-rises (slow)
2. **Peak vs. off-peak hours** — rush hour orders stack up, creating a second delay cluster
3. **Order complexity** — simple grocery packs vs. multi-item Fresh Food orders with different prep times

**Why This Must Be Addressed Before Modeling:**
The model would try to fit a single pattern across two fundamentally different sub-populations, producing inaccurate predictions for both.

**Modeling Mistake if Ignored:**
The model learns one decision boundary near the mean (~18 min). Orders in the 20–28 min range get systematically mislabeled — "on-time for Zone 2, late for Zone 1." This is an instance of **Simpson's Paradox** in aggregated hub data.

> ✅ **Fix:** Add `hub_zone_type` feature or cluster hubs by historical delivery speed profile before training.

---

## ⚖️ (e) Business Trade-Off — Precision vs Recall

### VP Operations' Insight (Kavya Sharma):
> *"30+ min delivery → churn increases. But aggressively flagging too many orders → unnecessary rider reallocations."*

| Scenario | Error Type | Cost |
|---|---|---|
| Miss a truly late delivery | False Negative | High — customer churns |
| Flag an on-time order as late | False Positive | Medium — wasted rider reallocation |

### Priority: **Recall** (with tiered response thresholds)

| Confidence Score | Action |
|---|---|
| > 0.80 | Immediate rider reallocation |
| 0.50 – 0.80 | Alert supervisor only |
| < 0.50 | No action |

> This balances recall (catching true late deliveries) with operational cost control (avoiding over-reallocation).

---

## 🏗️ (f) Feature Engineering

```python
# 1. Isolates last-mile delay vs. kitchen bottleneck
relative_delay_ratio = delivery_time_mins / prep_time_mins

# 2. Weights Fresh Food orders for higher complexity
order_complexity_score = items_count * (1 + (order_category == "Fresh Food"))

# 3. Flags weekday rush hours when hub throughput saturates
hub_peak_load_flag = 1 if (order_hour in [8,9,10,19,20,21]) and (is_weekend == 0) else 0
```

---

## 🛠️ Tech Stack (Recommended)

```
Language     : Python 3.10+
Libraries    : pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
EDA Tools    : pandas-profiling / ydata-profiling, scipy
Modeling     : XGBoost / LightGBM (Binary Classification)
Evaluation   : Recall, Precision, AUC-ROC, Confusion Matrix
```

---

## 📁 Repository Structure

```
zeptofresh-delivery-risk/
│
├── data/
│   └── orders_q2_2025_sample.csv
│
├── notebooks/
│   ├── 01_eda_observations.ipynb
│   ├── 02_data_quality_treatment.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training_evaluation.ipynb
│
├── src/
│   ├── eda_utils.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── outputs/
│   └── late_delivery_risk_scores.csv
│
└── README.md
```

---

## 📎 Submission

- Submit notebook + GitHub repository link on LMS
- Ensure all EDA cells are executed with charts visible
- Duration: 60–75 minutes

---

*IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering*
