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
