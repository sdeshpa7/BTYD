import pandas as pd
import numpy as np
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel, rfm_summary
from exploration import fit_models

def run_full_clv_prediction(data_path):
    # 1. Prepare Full Dataset RFM
    print("Loading full dataset and generating RFM summary...")
    raw_df = fit_models(data_path)
    
    rfm_full = rfm_summary(
        raw_df,
        customer_id_col='customer_id',
        datetime_col='visit_date',
        monetary_value_col='revenue',
    )
    
    # 2. Fit BG/NBD Model (Frequency) on Full Data
    print("\nFitting BG/NBD model on full dataset...")
    bgm = BetaGeoModel(data=rfm_full)
    bgm.build_model()
    bgm.fit(draws=2000, chains=4, target_accept=0.9, random_seed=42)
    
    # 3. Fit Gamma-Gamma Model (Monetary) on Full Data
    # We only fit on customers with frequency > 0
    repeat_customers = rfm_full[rfm_full["frequency"] > 0].copy()
    
    # Use the HalfNormal priors we established for stability
    gg_config = {
        "p_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
        "q_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
        "v_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 1000}},
    }
    
    print("\nFitting Gamma-Gamma model on full dataset...")
    ggm = GammaGammaModel(data=repeat_customers, model_config=gg_config)
    ggm.build_model()
    ggm.fit(draws=2000, chains=4, target_accept=0.95, random_seed=42)
    
    # 4. Generate Predictions
    print("\nGenerating 90-day and 365-day CLV predictions...")
    
    # A. Expected Number of Purchases (Frequency)
    # We predict for all customers
    freq_90 = bgm.expected_purchases(future_t=90, data=rfm_full).mean(dim=("chain", "draw")).to_series()
    freq_365 = bgm.expected_purchases(future_t=365, data=rfm_full).mean(dim=("chain", "draw")).to_series()
    
    # B. Expected Average Spend (Monetary)
    # Note: For new/one-time customers, we use the population average (model's intercept)
    # expected_customer_spend automatically handles this if we pass the whole rfm_full
    monetary_preds = ggm.expected_customer_spend(data=rfm_full).mean(dim=("chain", "draw")).to_series()
    
    # 5. Assemble Final Table
    results = rfm_full.copy()
    results['expected_purchases_90d'] = freq_90.values
    results['expected_purchases_365d'] = freq_365.values
    results['expected_avg_spend'] = monetary_preds.values
    
    # CLV Calculation: Frequency * Monetary
    results['clv_90d'] = results['expected_purchases_90d'] * results['expected_avg_spend']
    results['clv_365d'] = results['expected_purchases_365d'] * results['expected_avg_spend']
    
    # Sort by 365d CLV to find the MVPs
    results = results.sort_values('clv_365d', ascending=False)
    
    print("\n--- Final Customer Lifetime Value (CLV) Predictions ---")
    print(results[['customer_id', 'frequency', 'expected_avg_spend', 'clv_90d', 'clv_365d']].head(20))
    
    # Save results
    output_file = "final_clv_predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"\nFull predictions saved to {output_file}")
    
    # Save the full-data models as well
    bgm.save("bgm_full_model.nc")
    ggm.save("ggm_full_model.nc")
    print("Full-dataset models saved to .nc files.")

if __name__ == "__main__":
    run_full_clv_prediction("Ecommerce.csv")
