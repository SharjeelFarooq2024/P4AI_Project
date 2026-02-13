Confidence-Aware AI Inference Service
The deployed model outputs predictions with calibrated confidence and rejects or reroutes
low-confidence requests. Emphasis is on reliability, uncertainty handling, and API design.



Problem:
          Build an Intrusion Detection System (IDS) that classifies network traffic as malicious or benign and supports future confidence-aware inference for deployment.


JUSTIFICATION:

1️⃣ Realistic Cybersecurity Context

Simulates real modern network traffic

Contains multiple attack types

Designed for IDS research

2️⃣ Data Size Adequacy

~257k samples

49 network flow features

Suitable for:

Training

Validation

Calibration later

Error analysis

3️⃣ Feature Structure

Includes:

Numeric flow statistics

Categorical protocol features

Attack category label

Binary label

Supports both:

Binary IDS

Multi-class IDS

4️⃣ Future Compatibility

Supports:

Train/val/test split

Probability output

Threshold-based rejection

FastAPI deployment





Early Failures & Debugging section


Initial dataset loading resulted in malformed column structure due to absence of header rows in raw UNSW files. This led to inflated memory usage and incorrect feature count. The issue was resolved by explicitly loading feature names from the official feature file and assigning them during CSV import.
