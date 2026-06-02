import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymc_marketing import clv
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel, rfm_summary
from sklearn.model_selection import train_test_split
import arviz as az

from exploration import fit_models, create_RFM_Table

def run_gamma_gamma_workflow(bgm_model_path, raw_data_path):
    # 1. Full Dataset Analysis
    print(f"--- Phase 1: Full Dataset Analysis ---")
    data = fit_models(raw_data_path)
    data = data.sort_values('visit_date')
    train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)

    # Save the test data so we don't have to generate it again
    test_data.to_csv("test_data.csv", index=False)
    print("Holdout test data saved to 'test_data.csv'")

    rfm_full = create_RFM_Table(data, 'session_id', 'customer_id', 'visit_date', 'revenue', "False")
    repeat_full = rfm_full[rfm_full["Frequency Minus 1"] > 0].copy()
    
    corr_matrix_full = repeat_full[['Frequency', 'Average Monetary Value Per Order']].corr()
    print("\nFull Dataset Correlation Matrix:")
    print(corr_matrix_full)
    
    # 2. Phase 2: Model Fitting (Using EXACT Training Data from BG/NBD model)
    print(f"\n--- Phase 2: Model Fitting (Using training subset) ---")
    
    print(f"Loading BG/NBD model from {bgm_model_path} to extract training data...")
    bgm = BetaGeoModel.load(bgm_model_path)
    
    # Extract the exact RFM table used for the BG/NBD model
    rfm_train = bgm.data
    repeat_customers = rfm_train.query("frequency > 0").copy()
    
    print("\nTraining Set Correlation:")
    print(repeat_customers[['monetary_value', 'frequency']].corr())

    dataset = pd.DataFrame({
        'customer_id': repeat_customers['customer_id'],
        'monetary_value': repeat_customers['monetary_value'],
        'frequency': repeat_customers['frequency']
    })

    # 3. Initialize and Fit Gamma-Gamma Model with HalfNormal Priors
    # This prevents the parameters from sliding to infinity (the 10^15 issue)
    model_config = {
        "p_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
        "q_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 10}},
        "v_prior": {"dist": "HalfNormal", "kwargs": {"sigma": 1000}},
    }

    print("\nInitializing Gamma-Gamma model with HalfNormal priors...")
    gg = clv.GammaGammaModel(
        data=dataset,
        model_config=model_config
    )
    gg.build_model()

    sample_kwargs = {
        "draws": 2000,
        "chains": 4,
        "target_accept": 0.95,
        "random_seed": 42,
    }

    print("\nFitting Gamma-Gamma model on training data...")
    gg.fit(**sample_kwargs)
    
    print("\nGamma-Gamma Model Summary:")
    print(gg.fit_summary())
    
    # Save the model
    output_path = "ggm_model.nc"
    gg.save(output_path)
    print(f"\nGamma-Gamma model saved to {output_path}")

if __name__ == "__main__":
    run_gamma_gamma_workflow("bgm_model.nc", "Ecommerce.csv")
