# UNSW-NB15 Network Intrusion Detection System  
## Dataset Exploration & Initial Analysis (EDA)

**Project Documentation**  
**Date:** February 2026  
**Dataset:** UNSW-NB15 (University of New South Wales Network-Based Intrusion Detection Dataset)

---

## 1. Dataset Selection & Justification

For this project, three benchmark network intrusion detection datasets were considered:

1. **NSL-KDD**  
   - Created in 1999, widely used in academic research.  
   - Pros: Small and easy to handle, clean dataset.  
   - Cons: Outdated attack patterns, does not reflect modern network traffic.  

2. **CIC-IDS2017**  
   - Created in 2017, contains modern network traffic with realistic attacks.  
   - Pros: Recent and comprehensive dataset.  
   - Cons: Very large and complex feature space (80+ features with mixed types), heavy preprocessing, may be too complex for a semester project.  

3. **UNSW-NB15** (Chosen Dataset)  
   - Created in 2015 by the Australian Centre for Cyber Security (ACCS).  
   - Pros: Modern network traffic and attacks, manageable size (~2.5 million rows before sampling), mixed categorical and numerical features for realistic data engineering.  
   - Balances dataset complexity with feasibility for modular preprocessing, EDA, and baseline modeling.  
   - Provides an opportunity to explore memory optimization, duplicate handling, missing values, class imbalance, and feature correlations.

**Justification:** UNSW-NB15 provides a realistic and challenging dataset while remaining tractable for a semester-long project. NSL-KDD is outdated, and CIC-IDS2017 is too large and computationally demanding.

---

## 2. Executive Summary

This document presents the exploratory data analysis (EDA) of the UNSW-NB15 network intrusion detection dataset. The objective of this stage was to understand the dataset structure, data quality, feature distributions, and class imbalance before any preprocessing or modeling decisions.

The dataset consists of more than 2.5 million network flow records with 49 features. Through systematic EDA, several key issues were identified including duplicate records, missing values, memory constraints, and significant class imbalance.

---

## 3. Dataset Overview

### 3.1 Dataset Files Used

| File | Rows | Columns |
|------|------|---------|
| UNSW-NB15_1.csv | 700,001 | 49 |
| UNSW-NB15_2.csv | 700,001 | 49 |
| UNSW-NB15_3.csv | 700,001 | 49 |
| UNSW-NB15_4.csv | 440,044 | 49 |

**Total Combined Samples:** 2,540,047  
**Total Features:** 49  

---

### 3.2 Memory Usage

After merging all four CSV files:

- Total memory usage ≈ **2003.91 MB (~2GB)**  
- Large dataset size caused slow execution during initial exploration  

This highlighted the importance of memory awareness in handling large-scale tabular datasets.

---

### 3.3 Data Types

The dataset contains a mixture of:

- **Object (categorical) columns**  
  - srcip  
  - sport  
  - dstip  
  - dsport  
  - proto  
  - state  
  - service  
  - ct_ftp_cmd  
  - attack_cat  

- **Numerical columns (40 features)**  
  - dur  
  - sbytes  
  - dbytes  
  - sttl  
  - dttl  
  - Sload  
  - Dload  
  - Spkts  
  - Dpkts  
  - ct_srv_src  
  - ct_dst_src_ltm  
  - and others  

This confirms the dataset contains mixed feature types requiring careful handling.

---

## 4. Data Quality Analysis

### 4.1 Duplicate Records

EDA revealed:

- **480,632 duplicate rows**

This shows that the dataset contains redundant entries and is not fully cleaned.

---

### 4.2 Missing Values

Missing value analysis revealed:

| Column | Missing Values |
|--------|---------------|
| ct_flw_http_mthd | 1,348,145 |
| is_ftp_login | 1,429,879 |
| attack_cat | 2,218,764 |

This indicates that certain columns have a very high proportion of missing values and require special consideration in later preprocessing stages.

---

### 4.3 Infinite Values

- Number of infinite values: **0**

No infinite numerical values were detected.

---

## 5. Target Distribution Analysis

The dataset contains a binary target variable: **Label**

| Label | Count | Percentage |
|-------|--------|------------|
| 0 (Normal) | 2,218,764 | 87.35% |
| 1 (Attack) | 321,283 | 12.65% |

This reveals a significant class imbalance:

- Approximately **87% Normal traffic**
- Approximately **13% Attack traffic**

Such imbalance can bias downstream models if not handled properly.

A target distribution plot was generated and saved for visual confirmation.

---

## 6. Feature Type Analysis

- **Categorical Columns:**  
  srcip, sport, dstip, dsport, proto, state, service, ct_ftp_cmd, attack_cat  

- **Number of Numerical Columns:** 40  

Distribution plots were generated for key numerical features including:

- dur  
- sbytes  
- dbytes  
- sttl  
- dttl  

These plots helped visualize skewness and potential outliers in the data.

---

## 7. Correlation Analysis

A correlation heatmap was generated to examine relationships between numerical features.

This step helped identify:

- Highly correlated feature pairs  
- Redundant information  
- Potential multicollinearity  

The correlation matrix was saved for further analysis.

---

## 8. Key Observations from EDA

1. The dataset is large-scale (2,540,047 samples) and memory-intensive (~2GB).
2. A substantial number of duplicate rows (480,632) exist.
3. Some columns contain over 1 million missing values.
4. The dataset is highly imbalanced (87% vs 13%).
5. Multiple numerical features show skewed distributions.
6. Strong correlations exist between certain numerical variables.

These findings highlight that the dataset requires careful preprocessing before model training.

---

## 9.1 Preprocessing Pipeline

1. **Modular Design**  
   - Separate modules for data loading, feature engineering, encoding, scaling, and orchestration.  
   - Ensures reusability and maintainability for large-scale datasets.

2. **Initial Cleanup**  
   - Drop non-informative columns (e.g., `id`, `attack_cat`).  
   - Remove duplicate rows and handle missing values.

3. **Feature Engineering**  
   - Correlation analysis to remove highly correlated features.  
   - Mutual information to drop non-informative features.  
   - Outlier capping (Winsorization) at 1st and 99th percentiles to limit extreme values without removing critical attack patterns.

4. **Categorical Encoding**  
   - Hybrid encoding: Target encoding for high-cardinality features (`proto`), OneHot encoding for low-cardinality features (`service`, `state`).

5. **Numerical Scaling**  
   - Use `RobustScaler` to normalize features while being robust to outliers.

6. **Class Imbalance Handling**  
   - Apply SMOTE to oversample the minority class (Attack) for balanced model training.

7. **Pipeline Output**  
   - Save processed datasets and fitted preprocessing objects for reproducibility.

---

### 9.2 Model Training & Evaluation

1. **Model Selection**  
   - Random Forest Classifier chosen for robustness to outliers, interpretability via feature importance, and ability to handle non-linear relationships.

2. **Hyperparameters**  
   - Example: `n_estimators=100`, `max_depth=20`, `min_samples_split=5`, `class_weight='balanced'`.

3. **Training**  
   - Train on preprocessed and balanced dataset.  
   - Validate using the official test split.

4. **Evaluation Metrics**  
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC.  
   - Confusion matrix to check true positives and false negatives.  
   - Feature importance analysis to interpret predictive drivers.

---

### 9.3 Next Steps Summary

- Implement the full preprocessing pipeline based on the steps above.
- Train and evaluate the Random Forest model.  
- Use results to inform additional feature engineering or model selection if needed.  
- Prepare final dataset transformations, visualizations, and reports for project submission.

---

**Document End**  
*EDA Stage – Assignment 1 (Dataset Understanding)*
