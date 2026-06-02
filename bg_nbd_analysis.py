import pandas as pd
import matplotlib.pyplot as plt
from pymc_marketing import clv
from pymc_marketing.clv import BetaGeoModel
from exploration import fit_models, aggregate_by_time
import numpy as np
import arviz as az 
from arviz.labels import MapLabeller

def plot_probability_alive_matrix_custom(model):
    """Generates a masked Probability Alive matrix starting from Frequency=1"""
    print("Generating masked Probability Alive matrix...")
    max_freq = int(model.data["frequency"].max())
    max_rec = int(model.data["recency"].max())
    
    # Create grid of integers starting from 1 (excluding one-time buyers)
    freq_grid = np.arange(1, max_freq + 1)
    rec_grid = np.arange(0, max_rec + 1)
    
    # Create a dummy dataframe for all combinations
    ff, rr = np.meshgrid(freq_grid, rec_grid, indexing='ij')
    
    grid_df = pd.DataFrame({
        "customer_id": np.arange(ff.size),
        "frequency": ff.ravel(),
        "recency": rr.ravel(),
        "T": max_rec
    })
    
    # Calculate probability alive
    prob_alive = model.expected_probability_alive(data=grid_df)
    prob_values = prob_alive.mean(dim=("chain", "draw")).values.reshape(ff.shape)
    
    # MASK: Frequency > 0 and Recency = 0 is impossible
    mask = (ff > 0) & (rr == 0)
    prob_values = np.where(mask, np.nan, prob_values)
    
    # Plotting
    plt.figure(figsize=(12, 9))
    f_edges = np.arange(1, max_freq + 2) 
    r_edges = np.arange(0, max_rec + 2) 
    
    pcm = plt.pcolormesh(f_edges, r_edges, prob_values.T, cmap='viridis', shading='flat')
    plt.colorbar(pcm, label='P(Alive)')
    
    plt.xlabel('Frequency (Repeat Purchases)')
    plt.ylabel('Recency (Days between first and last purchase)')
    plt.title('Probability Alive Matrix (Frequency >= 1)')
    
    plt.xticks(np.arange(1, max_freq + 1))
    plt.yticks(np.arange(0, max_rec + 1, max(1, max_rec//10)))
    
    plt.xlim(1, max_freq + 1)
    plt.ylim(0, max_rec + 1)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

def plot_probability_alive_over_time(model, customer_id, raw_data, max_t=None):
    """
    Plots the HISTORICAL probability of being alive. 
    The probability jumps up when a purchase is made because the 
    customer has 'proven' they are alive at that moment.
    """
    # 1. Get transaction history from raw data
    cust_transactions = raw_data[raw_data["customer_id"] == customer_id].copy()
    if cust_transactions.empty:
        print(f"Error: Customer {customer_id} not found.")
        return
    
    cust_transactions = cust_transactions.sort_values("visit_date")
    first_purchase = cust_transactions["visit_date"].min()
    cust_transactions["days_since_birth"] = (cust_transactions["visit_date"] - first_purchase).dt.days
    purchase_dates = cust_transactions["days_since_birth"].values
    
    if max_t is None:
        max_t = int(model.data["T"].max())
        
    print(f"Generating jumping P(Alive) plot for Customer {customer_id}...")
    
    # 2. Build the historical timeline
    # For every day 't', we need the x and t_x observed *at that time*
    history_at_t = []
    t_timeline = np.arange(0, max_t + 1)
    
    for t in t_timeline:
        # Purchases that happened strictly BEFORE or ON day t
        past_purchases = purchase_dates[purchase_dates <= t]
        
        # In BTYD, x is REPEAT purchases (total - 1)
        x_at_t = max(0, len(past_purchases) - 1)
        # Recency is time of last purchase relative to first
        tx_at_t = past_purchases[-1] if len(past_purchases) > 0 else 0
        
        history_at_t.append({
            "customer_id": t, # Unique ID for each day's calculation
            "frequency": x_at_t,
            "recency": tx_at_t,
            "T": t
        })
    
    predict_df = pd.DataFrame(history_at_t)
    
    # 3. Calculate running probability
    prob_alive_samples = model.expected_probability_alive(data=predict_df)
    p_mean = prob_alive_samples.mean(dim=("chain", "draw")).values
    hdi_ds = az.hdi(prob_alive_samples, hdi_prob=0.95)
    var_name = list(hdi_ds.data_vars)[0]
    p_hdi = hdi_ds[var_name].values
    
    # 4. Plotting
    plt.figure(figsize=(14, 7))
    
    # Shaded HDI
    plt.fill_between(t_timeline, p_hdi[:, 0], p_hdi[:, 1], alpha=0.2, color='#3498db', label='95% Credible Interval')
    
    # The Jumping Line
    plt.plot(t_timeline, p_mean, color='#2980b9', lw=2.5, label='P(Alive) - Real-time')
    
    # Vertical Lines for Purchases
    for i, pt in enumerate(purchase_dates):
        label = "Purchase Event" if i == 0 else None
        plt.axvline(x=pt, color='black', linestyle='--', alpha=0.5, lw=1, label=label)
        # Small dots on the jumps
        plt.plot(pt, p_mean[pt], 'ko', markersize=4)

    plt.title(f'Historical Probability of Being Alive (Customer {customer_id})', fontsize=16, fontweight='bold')
    plt.xlabel('Days Since First Purchase (T)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left', frameon=True)
    plt.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.show()

def run_analysis(model_path):
    # 1. Load the pre-fitted model
    print(f"Loading model from {model_path}...")
    bgm_mcmc = BetaGeoModel.load(model_path)
    bgm_mcmc.build_model()
    print("Model loaded and rebuilt successfully.")


    # az.plot_posterior(bgm_mcmc.fit_result)

    # axes = az.plot_trace(
    # data=bgm_mcmc.idata,
    # compact=True,
    # kind="rank_bars",
    # backend_kwargs={"figsize": (12, 9), "layout": "constrained"},
    # )
    # plt.gcf().suptitle("BG/NBD Model Trace", fontsize=18, fontweight="bold");

    # clv.plot_frequency_recency_matrix(bgm_mcmc)
    # plt.show()
   

    # 1. Load the raw transaction data for event marking
    print("Loading raw transaction data...")
    raw_data = fit_models('Ecommerce.csv')

    # 2. Visualization
    # Pick a specific customer to showcase. 
    example_id = bgm_mcmc.data[bgm_mcmc.data["frequency"] > 1]["customer_id"].iloc[0]
    plot_probability_alive_over_time(bgm_mcmc, customer_id=example_id, raw_data=raw_data)





if __name__ == "__main__":
    try:
        run_analysis("bgm_model.nc")
    except FileNotFoundError:
        print("Error: 'bgm_model.nc' not found. Please run 'python3 btyd.py' first to fit and save the model.")
