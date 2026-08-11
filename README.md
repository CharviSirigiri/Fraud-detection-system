# Fraud-detection-system
# Financial Fraud Detection System

An end-to-end Machine Learning pipeline and interactive web application designed to detect fraudulent financial transactions using the PaySim synthetic dataset.

---

## 📌 Project Overview

Extreme class imbalance makes financial fraud detection challenging (~0.13% fraudulent cases). This project analyzes transaction patterns, engineers custom balance features, and trains a weighted classification model deployed via a **Streamlit** user interface.

---

## 📊 Dataset Summary

* **Source:** `AIML Dataset.csv` (PaySim Financial Dataset)
* **Total Transactions:** 6,362,620
* **Features:** 11 original columns including transaction type, amount, origin/destination balances, and fraud labels.
* **Class Imbalance:**
* **Legitimate (`0`):** 6,354,407 (99.87%)
* **Fraudulent (`1`):** 8,213 (0.13%)



---

## ⚙️ Key Findings & Feature Engineering

1. **High-Risk Transaction Types:** Fraud occurs exclusively in `TRANSFER` and `CASH_OUT` transaction types.
2. **Account Zeroing:** A significant number of fraudulent transactions deplete the origin account balance completely (`newbalanceOrig == 0`).
3. **Engineered Features:**
* `balanceDiffOrig = oldbalanceOrg - newbalanceOrig`
* `balanceDiffDest = newbalanceDest - oldbalanceDest`



---

## 🛠️ Model Architecture & Performance

The pipeline utilizes **Logistic Regression** with class weight balancing to handle severe label skew.

* **Preprocessing:** `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder(drop='first')` for categorical features.
* **Class Weighting:** `class_weight='balanced'`

### Evaluation Metrics (Test Set)

| Metric | Non-Fraud (`0`) | Fraud (`1`) | Overall / Macro Avg |
| --- | --- | --- | --- |
| **Precision** | 1.00 | 0.02 | 0.51 |
| **Recall** | 0.95 | **0.94** | 0.94 |
| **F1-Score** | 0.97 | 0.04 | 0.51 |
| **Accuracy** | — | — | **94.56%** |

> **Note:** The model prioritizes **high recall (94%)** on fraudulent transactions to minimize missed fraud cases (false negatives), accepting lower precision as an initial screening step.

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.8+

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn

```

### Step 2: Model Training

Train the model and save the pipeline artifact (`fraud_detection_model.pkl`):

```bash
python train_model.py

```

### Step 3: Run the Streamlit Web Application

```bash
streamlit run app.py

```

---

## 📱 Application Interface

The Streamlit web app takes transaction inputs, dynamically calculates engineered features, and returns a real-time risk assessment:

* **Legitimate Transaction:** Displays a green success banner.
* **Fraudulent Transaction:** Displays a red alert banner.
