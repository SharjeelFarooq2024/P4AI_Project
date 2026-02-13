# Dataset Selection and Justification

**Selected Dataset:** UNSW-NB15

**Reason for Selection:**
- Realistic modern network traffic generated with IXIA PerfectStorm, combining normal activity and contemporary attack types.
- Balanced attack types across 9 major categories (e.g., DoS, Exploits, Fuzzers, Reconnaissance) for multi-class and binary tasks.
- Rich 49-feature set spanning network-level, flow-level, and content-level attributes to enable diverse feature engineering.
- Large-scale (~2.5M samples) for training robust ML/DL models without severe overfitting.
- Community recognition as a widely cited benchmark for intrusion detection.

**Comparison with Other Common Datasets**

| Feature / Dataset      | UNSW-NB15                                 | CIC-IDS2017                                        | NSL-KDD                              |
|------------------------|-------------------------------------------|----------------------------------------------------|--------------------------------------|
| Year of Generation     | 2015                                      | 2017                                               | 1999                                 |
| Traffic Type           | Modern synthetic + realistic              | Modern simulated + realistic                       | Old simulated TCP/IP traffic         |
| Attack Types           | 9 categories (DoS, Exploit, etc.)         | 14 categories (DoS, DDoS, Brute Force, Botnet, etc.) | 4 main categories (DoS, Probe, U2R, R2L) |
| Number of Features     | 49 (network + flow + content)             | 80+ features (timestamps, protocols, flow stats)   | 41 (mostly network-level, no content-level) |
| Sample Size            | ~2.5 million                              | ~2.8 million                                       | 41K+ train, 12K test                 |
| Strengths              | Realistic modern attacks; well-structured | Very recent attacks; high fidelity; benign + attack flows | Widely used benchmark; lightweight   |
| Limitations            | Synthetic generation may miss real-world noise | Complex feature set; preprocessing intensive       | Outdated traffic; mostly old attacks |

**Justification Over Others:**
- Modern relevance: captures current attack patterns and protocols (unlike NSL-KDD).
- Manageable complexity: smaller yet sufficient feature set versus CIC-IDS2017, easing EDA and preprocessing.
- Suitable for binary and multi-class classification: labeled normal traffic plus diverse attack categories.

**Conclusion:** UNSW-NB15 balances realism, feature richness, and usability, making it a strong choice for an end-to-end ML intrusion detection project.

# Intrusion Detection EDA

## Dataset & Key Fields
- `sbytes` / `dbytes`: Source/ destination bytes transferred in a flow.
- `Spkts` / `Dpkts`: Source/ destination packet counts.
- `sttl` / `dttl`: Source/ destination IP time-to-live values observed.
- `Sload` / `Dload`: Source/ destination byte-throughput per second.
- `Sjit` / `Djit`: Source/ destination jitter (inter-arrival variation).
- `dur`: Flow duration (seconds).
- `proto`: Transport protocol (e.g., TCP, UDP, ICMP).
- `state`: Connection state label from Zeek/Argus (e.g., CON, FIN, REJ).
- `service`: Application layer service (e.g., http, ftp, dns). May be blank.
- `ct_*` features: Count-based context features (e.g., `ct_dst_ltm` = connections to same destination in a long-term window; `ct_src_dport_ltm` = connections from source to destination port in long-term window; `ct_state_ttl` = count of state/TTL pairs).
- `attack_cat`: Multi-class attack category (Normal, Fuzzers, DoS, Reconnaissance, etc.).
- `Label`: Binary target (0 = normal, 1 = attack). Note: some CSVs also include a lowercase `label`; we use the populated `Label` column.

## What the EDA Script Produces (`python main_eda.py`)
Artifacts are written under `figures/` and `figures/stats/`.

1) **Data overview**
	- Basic info, memory footprint, duplicate/missing/infinite checks.
	- Class distribution for `Label` (imbalance check).

2) **Numeric features**
	- Histograms (clipped at 99th percentile to make scales readable) for a few representative numeric columns.
	- IQR-based outlier summary CSV (`figures/stats/outlier_summary_iqr.csv`) plus boxplots for sample numeric columns (`figures/boxplots/`).

3) **Categorical features**
	- Value counts saved per column (`figures/stats/cat_counts_<col>.csv`).
	- Barplots of the top categories (`figures/categorical/`).
	- Target balance per category saved as CSV (`figures/stats/target_balance_<col>.csv`).

4) **Feature relationships**
	- Correlation heatmap (`figures/correlation_matrix.png`) on a sample of numeric rows.
	- Highly correlated pairs saved to `figures/stats/high_correlation_pairs.csv` (|r| >= 0.9) to flag potential redundancy.
	- Pairwise scatterplots for the most correlated numeric pairs (`figures/scatter/`), colored by `Label` when available.

## How to Run
```
python main_eda.py
```
Results print to console with the paths of saved plots/CSVs.

## Notes on Scale & Outliers
- Many traffic features are heavy-tailed (e.g., `dbytes`, `sbytes`), so raw histograms can look flat except near zero. We clip at the 99th percentile for plots to reveal the bulk of the data while keeping extremes in the outlier summary.
- Use the IQR summary to see how many rows fall outside typical ranges; adjust clipping or log-scale plots if you need more detail on tails.

## Assumptions vs. Reality (fill in after running)
- Expected: mix of normal and multiple attack categories with moderate imbalance; numeric features may be skewed; some services/protocols dominate.
- Verify after runs: class imbalance percentages, presence of heavy tails/zeros, missingness in `attack_cat` and other fields, any near-constant features. Record observations in your report under “Data Reality & Assumptions.”
