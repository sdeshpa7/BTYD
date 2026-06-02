# Customer Lifetime Value (CLV) with Buy 'Til You Die (BTYD) Models

This project implements and validates a Bayesian Buy 'Til You Die (BTYD) modeling pipeline using `pymc-marketing` to forecast customer transaction frequency, churn probability, and Customer Lifetime Value (CLV) in a non-contractual setting.

## Dataset
The project analyzes `Ecommerce.csv`, which contains transaction data from an Indian e-commerce brand sourced from Kaggle. The dataset contains transaction logs including customer IDs, visit dates, and revenue per order.

---

## Buy 'Til You Die (BTYD) Concepts

In non-contractual business settings (e.g., standard retail/e-commerce), customer churn is silent—customers simply stop buying without formal cancellation. BTYD models solve this by modeling transaction behavior and dropout as joint probability processes.

The framework consists of two separate models:

### 1. BG/NBD Model (Beta-Geometric / Negative Binomial Distribution)
The BG/NBD model predicts **how many purchases** a customer will make in a future time period and the probability that they are still active ("alive").

* **Transaction Process (NBD)**: While active, a customer's transactions follow a **Poisson process** with a purchase rate λ. Across the customer population, these transaction rates follow a **Gamma distribution**.
* **Dropout Process (Beta-Geometric)**: A customer can drop out (become permanently inactive) immediately after any purchase with a dropout probability p. Across the customer population, these dropout probabilities follow a **Beta distribution**.

### 2. Gamma-Gamma Model
The Gamma-Gamma model predicts the **expected average spend per transaction** (Average Order Value / AOV) for each customer.

* **Assumptions**: 
  * A customer's transaction value varies randomly around their personal mean transaction value.
  * The personal mean transaction values vary across the population according to a **Gamma distribution**.
  * **Independence Assumption**: The monetary value of transactions is assumed to be independent of transaction frequency. We verify this assumption by checking the correlation between frequency and average spend (correlation should be near 0).

### 3. Customer Lifetime Value (CLV) Calculation
Once both models are fitted, CLV for a future horizon t is calculated by combining their outputs:

`Expected CLV = Expected Transactions * Expected Average Spend`

### 4. Core Mathematical Formulations

#### Probability of Being Active: P(active)
The probability that a customer with a given transaction frequency, recency, and age is active (alive) at the end of the calibration period:

```
P(active | r, α, a, b, x, tₓ, T) = 1 / [ 1 + δ(x > 0) * (a / (b + x - 1)) * ((α + T) / (α + tₓ))^(r + x) ]
```
where:
* `δ(x > 0)` is an indicator function that equals 1 if the customer has repeat transactions (frequency > 0), and 0 otherwise.
* `frequency (x)` = number of repeat transactions.
* `recency (tₓ)` = time in days between the customer's first and last purchase.
* `age (T)` = total time observed since the customer's first purchase.
* `r, α, a, b` are the parameters of the BG/NBD model.

#### Expected Transactions: E[Y(t)]
The expected number of future transactions over a future time horizon `t` for a customer with history `(x, tₓ, T)`:

```
E[Y(t) | r, α, a, b, x, tₓ, T] = [ (a + b + x - 1) / (a - 1) ] * [ 1 - ((α + T) / (α + T + t))^(r + x) * F(r + x, b + x; a + b + x - 1; t / (α + T + t)) ] / [ 1 + δ(x > 0) * (a / (b + x - 1)) * ((α + T) / (α + tₓ))^(r + x) ]
```
where:
* `F` represents the Gauss Hypergeometric Function (implemented via `scipy.special.hyp2f1`).
* `t` = future time horizon.

#### Expected Average Spend: E(M)
The expected transaction size (monetary value) for a customer with a given historical average transaction size and transaction frequency:

```
E(M | p, q, v, mₓ, x) = (p * x * mₓ + v) / (p * x + q - 1)
```
where:
* `mₓ` = observed historical average spend.
* `p, q, v` are the parameters of the Gamma-Gamma model.
* This operates as a Bayesian shrinkage estimator, pulling individual values toward the population mean.

---

## Technical Workflow & Project Structure

The project code is divided into the following data science scripts:

* **[exploration.py](exploration.py)**: Loads raw transaction data, aggregates logs into RFM (Recency, Frequency, Age) statistics, validates assumptions, and splits the data chronologically into a **70% training (calibration)** and **30% validation (holdout)** split.
* **[btyd.py](btyd.py)**: Fits the BG/NBD frequency model on the training subset using PyMC Marketing. Saves the fitted model parameter trace to `bgm_model.nc`.
* **[bg_nbd_analysis.py](bg_nbd_analysis.py)**: Performs diagnostics and generates plots using the fitted BG/NBD model. Includes a masked Probability Alive matrix (excluding one-time buyers) and a real-time "jumping" probability alive plot over a customer's purchase timeline (tracking how P(Alive) decays between transactions and jumps back up immediately at transaction events).
* **[fit_btyd_model.py](fit_btyd_model.py)**: A helper script that fits both BG/NBD and Gamma-Gamma models in one go on repeat customers (frequency >= 2), scales monetary values by 100 for convergence stability, calculates 10/30/60-day expected CLV, and outputs a combined report to `rfm_with_predictions.csv` alongside model heatmaps.
* **[gamma-gamma.py](gamma-gamma.py)**: Fits the Gamma-Gamma monetary model on training data. Implements informative `HalfNormal` priors to ensure convergence and prevent parameter scaling issues. Saves the model trace to `ggm_model.nc`.
* **[gg-analysis.py](gg-analysis.py)**: Performs diagnostics and trace plots for the Gamma-Gamma monetary model fit results.
* **[predict_monetary.py](predict_monetary.py)**: Generates monetary validation reports on the holdout period. Groups customers by training spend deciles and plots a comparison of historical calibration spend, actual holdout test spend, and predicted average spend to validate monetary estimates.
* **[frequency_validation.py](frequency_validation.py)**: Performs empirical calibration validation of the BG/NBD frequency predictions against the 109-day holdout dataset. Saves the calibration plot to `frequency_validation.png`.
* **[clv_prediction_full.py](clv_prediction_full.py)**: Production script that fits both models on the complete dataset (using the established informative priors) to generate 90-day and 365-day CLV forecasts, outputting findings to `final_clv_predictions.csv` and full models as `.nc` files.
* **[illustrate_priors.py](illustrate_priors.py)**: Generates a plot comparing the informative `HalfNormal` priors vs. uninformative flat priors to document the Bayesian modeling choices.
* **[verify_math.py](verify_math.py)**: Manually computes the closed-form BG/NBD expectation equation using Gauss Hypergeometric functions and compares it to the library's output to verify mathematical consistency.

---

## Validation Strategy

To ensure model predictions are reliable before deploying to production, two validation pipelines are implemented:

### 1. Empirical Holdout Validation
We evaluate predictive performance by checking how well models trained on the first 70% of the timeline predict behavior in the remaining 30% holdout period:
* **Frequency Validation**:
  * We calculate the actual number of transactions completed by each customer during the holdout period.
  * We predict the expected transactions for the same holdout window size.
  * We generate a **Calibration Plot** by grouping customers by their training-period purchase frequency (e.g., 1, 2, 3, 4+ purchases) and plotting the average actual transactions vs. the average model-predicted transactions for each tier. This checks if the model accurately scales predictions across different customer segments.
* **Monetary Validation**:
  * We compare predicted average spend against actual average spend in the test set.

### 2. Mathematical Verification
* **[verify_math.py](verify_math.py)**: Contains a script to manually compute the exact closed-form BG/NBD expected transaction equation (incorporating the **Gauss Hypergeometric Function** F(a, b; c; z)) using `scipy.special.hyp2f1`. 
* It compares manual evaluations against the `expected_purchases` outputs from `pymc-marketing` for specific customers to verify mathematical logic and sampling code correctness.

---

## Model Parameters and Results

The Bayesian MCMC parameter estimates (posterior means) for the models fit on the complete Indian e-commerce dataset:

### 1. BG/NBD Model Parameters (Transaction & Churn)
| Parameter | Description | Posterior Mean |
| :--- | :--- | :--- |
| r | Transaction shape parameter | 0.277 |
| alpha | Transaction scale parameter | 82.826 |
| a | Churn shape parameter (Beta distribution shape 1) | 133.256 |
| b | Churn shape parameter (Beta distribution shape 2) | 100.993 |

### 2. Gamma-Gamma Model Parameters (Spend/Monetary Value)
| Parameter | Description | Posterior Mean |
| :--- | :--- | :--- |
| p | Monetary value shape parameter 1 | 2.163 |
| q | Monetary value shape parameter 2 | 5.605 |
| v | Monetary value scale parameter | 3798.341 |

### 3. Top Predicted Customers (365-Day CLV Forecasts)
Top customers sorted by predicted 365-day Customer Lifetime Value (CLV) in `final_clv_predictions.csv`:

| Customer ID | Historical Frequency | Predicted Avg Spend (AOV) | Expected Purchases (90d) | Expected Purchases (365d) | P(alive) | CLV (365d) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5176** | 3.0 | 3840.51 | 0.31 | 0.51 | 27.66% | 1943.90 |
| **7701** | 2.0 | 2290.94 | 0.43 | 0.74 | 44.29% | 1698.34 |
| **5376** | 1.0 | 3052.88 | 0.29 | 0.55 | 39.64% | 1692.33 |
| **7839** | 2.0 | 2695.32 | 0.28 | 0.58 | 38.87% | 1568.49 |
| **2939** | 4.0 | 2280.35 | 0.30 | 0.69 | 44.34% | 1566.72 |
| **9476** | 3.0 | 2075.76 | 0.39 | 0.75 | 44.50% | 1562.14 |

---

## Validation Performance Results

### 1. Empirical Holdout Validation (BG/NBD)
* **Holdout Horizon**: 109 days (30% temporal split)
* **Actual Transactions**: 599 total purchases
* **Predicted Transactions**: 344 total purchases
* **Overall Forecast Accuracy**: **57.40%**
* **Calibration Cohort Performance** (Average actual vs. predicted transactions in holdout period based on historical purchases):
  * **1.0 Historical Purchases**: 0.18 actual vs. 0.10 predicted
  * **2.0 Historical Purchases**: 0.24 actual vs. 0.16 predicted
  * **3.0 Historical Purchases**: 0.20 actual vs. 0.24 predicted
  * **4.0 Historical Purchases**: 0.00 actual vs. 0.31 predicted (Note: small cohort size of only 3 customers who happened not to purchase in the holdout period)

### 2. Mathematical Exactness Verification (`verify_math.py`)
Tested for Customer `1334` predicting transactions over a 365-day horizon:
* **Manual Calculation (Gauss Hypergeometric implementation)**: 0.233316
* **Library output (`pymc-marketing`)**: 0.233316
* **Absolute Difference**: 0.000000 (Exact match, validating mathematical implementation correctness)

