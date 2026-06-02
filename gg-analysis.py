import pandas as pd
import matplotlib.pyplot as plt
from pymc_marketing import clv
from pymc_marketing.clv import GammaGammaModel
from exploration import fit_models, aggregate_by_time
import numpy as np
import arviz as az 
from arviz.labels import MapLabeller

def run_analysis(model_path):
    # 1. Load the pre-fitted model
    print(f"Loading model from {model_path}...")
    gg = GammaGammaModel.load(model_path)
    gg.build_model()
    print("Model loaded and rebuilt successfully.")

    # az.plot_posterior(gg.fit_result)

    # axes = az.plot_trace(
    #     data=gg.idata,
    #     compact=True,
    #     kind="rank_bars",
    #     backend_kwargs={"figsize": (12, 9), "layout": "constrained"},
    # )
    # plt.gcf().suptitle("GG Model Trace", fontsize=18, fontweight="bold");

    plt.show()

if __name__ == "__main__":

    try:
        run_analysis("ggm_model.nc")
    except FileNotFoundError:
        print("Error: 'ggm_model.nc' not found. Please run 'python3 gamma-gamma.py' first to fit and save the model.")

    