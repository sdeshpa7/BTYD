import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymc_marketing.clv import BetaGeoModel

def run_frequency_validation(model_path, test_data_path):
    # 1. Load the Frequency Model and Holdout Data
    print(f"Loading BG/NBD model from {model_path}...")
    bgm = BetaGeoModel.load(model_path)
    bgm.build_model()
    
    print(f"Loading holdout data from {test_data_path}...")
    test_data = pd.read_csv(test_data_path)
    test_data['visit_date'] = pd.to_datetime(test_data['visit_date'])
    
    # 2. Calculate Holdout Period Duration (T_holdout)
    # We need to know how many days to predict for
    holdout_start = test_data['visit_date'].min()
    holdout_end = test_data['visit_date'].max()
    t_holdout = (holdout_end - holdout_start).days
    print(f"Holdout period duration: {t_holdout} days")
    
    # 3. Actual Transactions in Holdout
    # We count unique dates per customer as a 'transaction'
    actual_counts = test_data.groupby('customer_id')['visit_date'].nunique().reset_index()
    actual_counts.columns = ['customer_id', 'actual_transactions']
    
    # 4. Predict Transactions for the Holdout Period
    # Using the training data already in the model
    print(f"Predicting transactions for the next {t_holdout} days...")
    predictions = bgm.expected_purchases(
        future_t=t_holdout,
        data=bgm.data
    ).mean(dim=("chain", "draw")).to_dataframe(name="predicted_transactions").reset_index()
    
    # 5. Merge and Compare
    # We want to see results for everyone who was in the training set
    comparison = pd.merge(bgm.data[['customer_id']], predictions, on='customer_id', how='left')
    comparison = pd.merge(comparison, actual_counts, on='customer_id', how='left').fillna(0)
    
    print("\n--- Frequency Validation (Holdout Period) ---")
    print(comparison.head(15))
    
    # Summary Metrics
    total_actual = comparison['actual_transactions'].sum()
    total_predicted = comparison['predicted_transactions'].sum()
    print(f"\nTotal Actual Transactions: {total_actual:.0f}")
    print(f"Total Predicted Transactions: {total_predicted:.0f}")
    print(f"Overall Accuracy: {min(total_actual, total_predicted) / max(total_actual, total_predicted):.2%}")

    # 5. Prepare Data for Binned Averages (Calibration Plot)
    calib_plot = pd.merge(comparison, bgm.data[['customer_id', 'frequency']], on='customer_id')
    
    # Filter for frequency >= 1 as seen in the screenshot and group up to 9
    calib_plot = calib_plot[calib_plot['frequency'] >= 1].copy()
    calib_plot['frequency_bin'] = calib_plot['frequency'].clip(upper=9)
    
    # Calculate Mean of actual and predicted for each frequency bin
    calib_grouped = calib_plot.groupby('frequency_bin', observed=True).agg({
        'actual_transactions': 'mean',
        'predicted_transactions': 'mean'
    })
    
    # Plotting
    plt.figure(figsize=(12, 7))
    
    x_axis = np.arange(len(calib_grouped))
    width = 0.35
    
    # Match the vibrant standard colors: Blue for Actual, Orange for Model
    plt.bar(x_axis - width/2, calib_grouped['actual_transactions'], width, 
            label='Actual', color='#3498db')
    plt.bar(x_axis + width/2, calib_grouped['predicted_transactions'], width, 
            label='Model', color='#e67e22')
    
    # Aesthetics
    plt.title('Holdout Period Transactions by Calibration Frequency', fontsize=15, fontweight='bold')
    plt.xlabel('Calibration Period Transactions', fontsize=12)
    plt.ylabel('Holdout Period Transactions (Average)', fontsize=12)
    
    plt.xticks(x_axis, calib_grouped.index.astype(int))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("frequency_validation.png")
    print("Plot saved as frequency_validation.png")

if __name__ == "__main__":
    run_frequency_validation("bgm_model.nc", "test_data.csv")
