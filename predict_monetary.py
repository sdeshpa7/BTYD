import pandas as pd
import matplotlib.pyplot as plt
from pymc_marketing.clv import GammaGammaModel
from exploration import fit_models
from sklearn.model_selection import train_test_split
import numpy as np

def run_holdout_prediction(model_path, test_data_path):
    # 1. Load the model
    print(f"Loading Gamma-Gamma model from {model_path}...")
    gg = GammaGammaModel.load(model_path)
    gg.build_model()
    
    # 2. Load the Pre-Saved Test Data (the holdout period)
    print(f"Loading holdout set from {test_data_path}...")
    test_data = pd.read_csv(test_data_path)
    
    # 3. Calculate Actual average spend for customers in the test period
    actual_spend = test_data.groupby("customer_id")["revenue"].mean().reset_index()
    actual_spend.columns = ["customer_id", "actual_avg_spend_test"]
    
    # 4. Predict Expected average spend using the model
    print("Generating predictions based on training history...")
    predictions = gg.expected_customer_spend(
        data=gg.data
    ).mean(dim=("chain", "draw")).to_dataframe(name="predicted_avg_spend").reset_index()
    
    # The xarray coordinates might name the index 'customer_id'
    # Let's ensure it's named correctly for the merge
    if "customer_id" not in predictions.columns and predictions.index.name == "customer_id":
        predictions = predictions.reset_index()
    
    # 5. Merge Calibration (Training), Holdout (Test), and Predictions
    # We need 'monetary_value' from the training data (Calibration)
    calibration_data = gg.data[["customer_id", "monetary_value"]].rename(columns={"monetary_value": "calibration_spend"})
    
    validation_df = pd.merge(actual_spend, predictions, on="customer_id", how="inner")
    validation_df = pd.merge(validation_df, calibration_data, on="customer_id", how="inner")
    
    print("\n--- Calibration vs Holdout vs Prediction ---")
    print(validation_df.head(10))
    
    # 6. Binned Comparison Plot
    # We group customers by their Calibration Spend into 5-10 bins (deciles)
    validation_df['calibration_bin'] = pd.qcut(validation_df['calibration_spend'], q=5, duplicates='drop')
    
    binned_data = validation_df.groupby('calibration_bin', observed=True).agg({
        'calibration_spend': 'mean',
        'actual_avg_spend_test': 'mean',
        'predicted_avg_spend': 'mean'
    }).reset_index()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    x_axis = np.arange(len(binned_data))
    width = 0.25
    
    plt.bar(x_axis - width, binned_data['calibration_spend'], width, label='Actual Calibration (Training)', color='#bdc3c7')
    plt.bar(x_axis, binned_data['actual_avg_spend_test'], width, label='Actual Holdout (Test)', color='#2980b9')
    plt.bar(x_axis + width, binned_data['predicted_avg_spend'], width, label='Predicted (Model)', color='#27ae60')
    
    plt.xticks(x_axis, [f"Group {i+1}" for i in range(len(binned_data))])
    plt.title('Monetary Value Validation: Calibration vs Holdout vs Predicted', fontsize=15, fontweight='bold')
    plt.xlabel('Customer Spending Groups (Binned by Calibration Spend)', fontsize=12)
    plt.ylabel('Average Transaction Value', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels on top of the bars
    for i in x_axis:
        plt.text(i, binned_data['actual_avg_spend_test'].iloc[i] + 10, f"{binned_data['actual_avg_spend_test'].iloc[i]:.0f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_holdout_prediction("ggm_model.nc", "test_data.csv")
