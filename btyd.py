import pandas as pd
import numpy as np
import pymc as pm
from pymc_marketing import clv
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel, rfm_summary
import matplotlib.pyplot as plt
import arviz as az
from sklearn.model_selection import train_test_split
from exploration import *



if __name__ == "__main__":


    data = fit_models('Ecommerce.csv')  
   

    #Test-Train Split

    data = data.sort_values('visit_date')

    train_data, test_data = train_test_split(data,test_size=0.3,shuffle=False)

    train_min_date=min(train_data['visit_date'])
    train_max_date = max(train_data['visit_date'])

    # print("Training Data Max Date: "+str(train_max_date))
    # print("Training Data Min Date: "+str(train_min_date))
    # print("Training Data Total Dur Days: " +str((train_max_date-train_min_date).days))
    # print("Training Data Total Dur Weeks: " +str(round((train_max_date-train_min_date).days/7,1)))
    # print("Training Data Total Dur Mths: " +str(round((train_max_date-train_min_date).days/30.417,1)))
    
    # print(train_data.describe())

    pymc_mktg_rfm_data = rfm_summary(
        train_data,
        customer_id_col='customer_id',
        datetime_col='visit_date',
        monetary_value_col='revenue',
    )
    
    pymc_mktg_rfm_data = pymc_mktg_rfm_data.sort_values("frequency",ascending=False)
    # print(pymc_mktg_rfm_data.head())


    # BGNBD Model
    bgm_mcmc = BetaGeoModel(
        data=pymc_mktg_rfm_data
    )
    
    bgm_mcmc.build_model()
    
    sample_kwargs = {
        "draws": 2_000,
        "chains": 4,
        "target_accept": 0.9,
        "random_seed": 42,
    }
    
    print("Fitting BG/NBD model...")
    bgm_mcmc.fit(**sample_kwargs)
    
    print(" Fitted Model Summary ")
    print(bgm_mcmc.fit_summary())
    
    # Save the model
    model_path = "bgm_model.nc"
    bgm_mcmc.save(model_path)
    print(f"Model saved to {model_path}")
