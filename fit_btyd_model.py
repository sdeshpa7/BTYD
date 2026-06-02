import pandas as pd
import pymc as pm
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel, rfm_summary
import matplotlib.pyplot as plt
import arviz as az

def fit_models(file_path):
    # 1. Load and Preprocess Data
    df = pd.read_csv(file_path)
    df_purchases = df[df['purchased'] == 1].copy()
    df_purchases['visit_date'] = pd.to_datetime(df_purchases['visit_date'], format='%d-%m-%Y')
    
    rfm = rfm_summary(
        df_purchases,
        customer_id_col='customer_id',
        datetime_col='visit_date',
        monetary_value_col='revenue',
    )
    
    # Exclude non-repeat customers (frequency < 2)
    rfm = rfm[rfm['frequency'] >= 2].copy()
    print(f"RFM summary filtered to {len(rfm)} repeat customers (freq >= 2).")
    
    # Normalization: Divide monetary value by 100 to help convergence
    rfm['monetary_value_original'] = rfm['monetary_value']
    rfm['monetary_value'] = rfm['monetary_value'] / 100.0
    
    # 2. Fit BG/NBD Model (Frequency/Dropout)
    print("Fitting BG/NBD Model...")
    bgm = BetaGeoModel(
        data=rfm,
    )
    bgm.fit(tune=2000, draws=2000, target_accept=0.9)
    print("BG/NBD Model fitted.")
    
    # 3. Fit Gamma-Gamma Model (Monetary Value)
    print(f"Fitting Gamma-Gamma Model on {len(rfm)} repeat customers...")
    
    ggm = GammaGammaModel(
        data=rfm,
    )
    ggm.fit(tune=2000, draws=2000, target_accept=0.95)
    print("Gamma-Gamma Model fitted.")
    
    return rfm, bgm, ggm

def analyze_results(rfm, bgm, ggm):
    # Print model summaries for diagnostics
    print("\nBG/NBD Model Fit Summary:")
    print(bgm.fit_summary())
    print("\nGamma-Gamma Model Fit Summary:")
    print(ggm.fit_summary())

    # 1. Predict average monetary value (Individual-level)
    print("Calculating individual expected monetary values...")
    expected_monetary_value = ggm.expected_customer_spend(data=rfm)
    # Scale back by 100
    rfm['expected_monetary_value'] = expected_monetary_value.mean(dim=("chain", "draw")).values * 100.0
    rfm['monetary_value'] = rfm['monetary_value_original'] # Restore for output

    # 2. Predict future transactions and CLV for different horizons
    horizons = [10, 30, 60]
    
    for t in horizons:
        col_purchases = f'expected_purchases_{t}d'
        col_clv = f'clv_{t}d'
        
        # Expected number of purchases in the next t days
        expected_purchases = bgm.expected_purchases(
            data=rfm,
            future_t=t
        )
        rfm[col_purchases] = expected_purchases.mean(dim=("chain", "draw")).values
        
        # Calculate CLV = Expected Purchases * Expected Monetary Value
        # Note: In a more advanced setup, we could include a discount rate here.
        rfm[col_clv] = rfm[col_purchases] * rfm['expected_monetary_value']
    
    # 3. Add 'Total Purchases' for clarity to the user
    # frequency is repeat purchases, so total = frequency + 1
    rfm['total_purchases'] = rfm['frequency'] + 1
    
    print("\nTop 10 Customers by Predicted 60-day CLV:")
    cols_to_show = ['customer_id', 'total_purchases', 'frequency', 'recency', 'T', 
                    'monetary_value', 'expected_monetary_value', 'clv_60d']
    print(rfm.sort_values('clv_60d', ascending=False)[cols_to_show].head(10))
    
    # Save results
    rfm.to_csv('rfm_with_predictions.csv', index=False)
    print("\nPredictions saved to rfm_with_predictions.csv")

    # 4. Visualizations
    print("Generating BTYD visualizations...")
    from pymc_marketing.clv import plot_frequency_recency_matrix, plot_probability_alive_matrix
    
    # Frequency-Recency Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_frequency_recency_matrix(bgm)
    plt.title("Expected Transactions in Next Period")
    plt.savefig('frequency_recency_heatmap.png')
    plt.close()
    
    # Probability Alive Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_probability_alive_matrix(bgm)
    plt.title("Probability Customer is Alive")
    plt.savefig('probability_alive_heatmap.png')
    plt.close()
    print("Visualizations saved: frequency_recency_heatmap.png, probability_alive_heatmap.png")

if __name__ == "__main__":
    rfm, bgm, ggm = fit_models('Ecommerce.csv')
    analyze_results(rfm, bgm, ggm)
    
    # Optional: Plot posteriors
    print("Saving posterior plots...")
    az.plot_posterior(bgm.idata)
    plt.savefig('bgm_posterior.png')
    plt.close()
    
    az.plot_posterior(ggm.idata)
    plt.savefig('ggm_posterior.png')
    plt.close()
    print("Plots saved.")
