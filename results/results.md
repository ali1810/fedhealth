# Results

5 simulated hospitals, 30 rounds, 5 seeds (mean ± std). Global test = held-out 20% of patients no hospital trained on.


## strong non-IID (α=0.3)

| Method | Global accuracy % | Global ROC AUC % | Accuracy on own hospital test % | ε (δ=1e-5) | Traffic (MB) | Time (s) |
|---|---|---|---|---|---|---|
| Centralised (pooled data) | 97.7 ± 1.1 | 99.4 ± 0.7 | - | - | - | - |
| Local only (mean of hospitals) | 71.6 ± 15.4 | 67.9 ± 26.1 | - | - | - | - |
| FedAvg | 97.0 ± 0.4 | 99.6 ± 0.2 | 96.9 ± 1.7 | ∞ (no DP) | 2.50 | 1.8 |
| FedAvg + SecAgg | 97.0 ± 0.4 | 99.6 ± 0.2 | 96.9 ± 1.7 | ∞ (no DP) | 3.77 | 2.0 |
| FedProx + SecAgg | 97.0 ± 0.4 | 99.6 ± 0.2 | 96.7 ± 1.4 | ∞ (no DP) | 3.77 | 3.2 |
| DP-FedAvg + SecAgg (z=0.5) | 89.5 ± 3.2 | 96.0 ± 1.6 | 92.9 ± 2.5 | 112.6 | 3.77 | 2.1 |
| DP-FedAvg + SecAgg (z=1.0) | 87.9 ± 5.0 | 94.8 ± 1.6 | 90.8 ± 5.3 | 41.3 | 3.77 | 2.2 |
| DP-FedAvg + SecAgg (z=2.0) | 83.0 ± 9.3 | 87.6 ± 4.8 | 84.6 ± 6.0 | 16.9 | 3.77 | 2.2 |

## moderate (α=1)

| Method | Global accuracy % | Global ROC AUC % | Accuracy on own hospital test % | ε (δ=1e-5) | Traffic (MB) | Time (s) |
|---|---|---|---|---|---|---|
| Centralised (pooled data) | 97.9 ± 0.9 | 99.6 ± 0.2 | - | - | - | - |
| Local only (mean of hospitals) | 86.0 ± 6.3 | 91.2 ± 8.3 | - | - | - | - |
| FedAvg | 97.2 ± 0.7 | 99.6 ± 0.2 | 97.3 ± 1.2 | ∞ (no DP) | 2.50 | 2.0 |
| FedAvg + SecAgg | 97.2 ± 0.7 | 99.6 ± 0.2 | 97.3 ± 1.2 | ∞ (no DP) | 3.77 | 2.2 |
| FedProx + SecAgg | 97.0 ± 0.7 | 99.6 ± 0.2 | 97.3 ± 1.2 | ∞ (no DP) | 3.77 | 3.5 |
| DP-FedAvg + SecAgg (z=0.5) | 90.9 ± 3.3 | 96.9 ± 1.5 | 91.6 ± 4.8 | 112.6 | 3.77 | 2.1 |
| DP-FedAvg + SecAgg (z=1.0) | 88.4 ± 4.5 | 95.8 ± 1.5 | 90.9 ± 6.3 | 41.3 | 3.77 | 2.2 |
| DP-FedAvg + SecAgg (z=2.0) | 86.3 ± 4.7 | 88.5 ± 3.2 | 88.6 ± 6.4 | 16.9 | 3.77 | 2.2 |

## near IID (α=100)

| Method | Global accuracy % | Global ROC AUC % | Accuracy on own hospital test % | ε (δ=1e-5) | Traffic (MB) | Time (s) |
|---|---|---|---|---|---|---|
| Centralised (pooled data) | 97.4 ± 0.8 | 99.5 ± 0.4 | - | - | - | - |
| Local only (mean of hospitals) | 96.0 ± 1.7 | 99.1 ± 0.8 | - | - | - | - |
| FedAvg | 97.2 ± 0.4 | 99.6 ± 0.2 | 97.0 ± 1.6 | ∞ (no DP) | 2.50 | 2.2 |
| FedAvg + SecAgg | 97.2 ± 0.4 | 99.6 ± 0.2 | 97.0 ± 1.6 | ∞ (no DP) | 3.77 | 2.3 |
| FedProx + SecAgg | 97.2 ± 0.4 | 99.6 ± 0.2 | 97.0 ± 1.6 | ∞ (no DP) | 3.77 | 3.7 |
| DP-FedAvg + SecAgg (z=0.5) | 92.8 ± 2.7 | 97.0 ± 1.8 | 92.0 ± 3.3 | 112.6 | 3.77 | 2.4 |
| DP-FedAvg + SecAgg (z=1.0) | 93.0 ± 2.7 | 95.9 ± 1.8 | 92.0 ± 1.9 | 41.3 | 3.77 | 2.4 |
| DP-FedAvg + SecAgg (z=2.0) | 89.6 ± 3.3 | 91.6 ± 2.1 | 90.8 ± 2.1 | 16.9 | 3.77 | 2.3 |

## Data quality reports (strong non-IID split, seed 0)

| Hospital | Training samples | Positive rate | Missing | Outlier share | Warnings |
|---|---|---|---|---|---|
| hospital_1 | 43 | 0.00 | 0.00 | 0.009 | small sample size, strong class imbalance |
| hospital_2 | 61 | 0.00 | 0.00 | 0.009 | strong class imbalance |
| hospital_3 | 192 | 1.00 | 0.00 | 0.011 | strong class imbalance |
| hospital_4 | 26 | 0.81 | 0.00 | 0.047 | small sample size, many extreme values |
| hospital_5 | 18 | 0.00 | 0.00 | 0.019 | small sample size, strong class imbalance |
