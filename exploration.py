import pandas as pd
import numpy as np
import pymc as pm
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel, rfm_summary
import matplotlib.pyplot as plt
import arviz as az
from sklearn.model_selection import train_test_split

def aggregate_by_time(data, order_id,customer_id,
                      timestamp, monetary_value, granularity="D"):
    """Aggregate transaction data to the customer level, computing RFM metrics"""
    df_monetary_value = data[[timestamp,customer_id,monetary_value]].groupby(
        [pd.Grouper(key=timestamp,freq=granularity),customer_id]).sum()

    df_monetary_value = df_monetary_value.reset_index()

    df_invoice_count = data[[timestamp,order_id,customer_id]].groupby([
        pd.Grouper(key=timestamp,freq=granularity),customer_id]).nunique()
    
    df_invoice_count = df_invoice_count.reset_index()
    
    df = df_monetary_value.merge(df_invoice_count, on = [timestamp,customer_id])

    df = df.rename(columns={order_id:'Unique Count of Order ID'})

    return df
    

def create_RFM_Table(data,order_id,customer_id,timestamp,monetary_value,granularity):
    
    max_date=data[timestamp].max()
    latest_order_date = pd.DataFrame(data.groupby([customer_id])[timestamp].max())
    latest_order_date.rename(columns = {timestamp:'Most Recent Order Date'}, inplace = True)
    latest_order_date["customer_unique_id"]=latest_order_date.index

    earliest_order_date=pd.DataFrame(data.groupby([customer_id])[timestamp].min())
    earliest_order_date.rename(columns = {timestamp:'Earliest Order Date'}, inplace = True)
    earliest_order_date["Age"]=(max_date-earliest_order_date["Earliest Order Date"]).dt.round("d")
    earliest_order_date['Age In Days'] = earliest_order_date['Age'] / pd.to_timedelta(1, unit='D')
    earliest_order_date.drop(['Age'], axis=1, inplace=True)    
    earliest_order_date["customer_unique_id"]=earliest_order_date.index

    recency_df=pd.merge(latest_order_date[["Most Recent Order Date"]],
                            earliest_order_date[["Earliest Order Date","Age In Days"]],
                            left_index=True, right_index=True)

    recency_df["recency"]=recency_df["Most Recent Order Date"]-recency_df["Earliest Order Date"]

    recency_df["recency"]=recency_df["recency"].dt.round("d")
    recency_df.rename(columns = {"recency":"Recency In Days"}, inplace = True)
    recency_df['Recency In Days'] = recency_df['Recency In Days'] / pd.to_timedelta(1, unit='D')
    recency_df[customer_id]=recency_df.index
    recency_df.reset_index(inplace = True, drop = True)


def fit_models(file_path):
    # 1. Load and Preprocess Data
    df = pd.read_csv(file_path)
    df_purchases = df[df['purchased'] == 1].copy()
    df_purchases['visit_date'] = pd.to_datetime(df_purchases['visit_date'], format='%d-%m-%Y')
    return df_purchases


def create_RFM_Table(data, order_id, customer_id, timestamp, monetary_value, granularity):
    max_date = data[timestamp].max()
    latest_order_date = pd.DataFrame(data.groupby([customer_id])[timestamp].max())
    latest_order_date.rename(columns={timestamp: 'Most Recent Order Date'}, inplace=True)

    earliest_order_date = pd.DataFrame(data.groupby([customer_id])[timestamp].min())
    earliest_order_date.rename(columns={timestamp: 'Earliest Order Date'}, inplace=True)
    earliest_order_date["Age"] = (max_date - earliest_order_date["Earliest Order Date"]).dt.round("D")
    earliest_order_date['Age In Days'] = earliest_order_date['Age'] / pd.to_timedelta(1, unit='D')
    earliest_order_date.drop(['Age'], axis=1, inplace=True)

    recency_df = pd.merge(latest_order_date[["Most Recent Order Date"]],
                          earliest_order_date[["Earliest Order Date", "Age In Days"]],
                          left_index=True, right_index=True)

    recency_df["recency"] = recency_df["Most Recent Order Date"] - recency_df["Earliest Order Date"]
    recency_df["recency"] = recency_df["recency"].dt.round("D")
    recency_df.rename(columns={"recency": "Recency In Days"}, inplace=True)
    recency_df['Recency In Days'] = recency_df['Recency In Days'] / pd.to_timedelta(1, unit='D')
    recency_df[customer_id] = recency_df.index
    recency_df.reset_index(inplace=True, drop=True)

    # Use a copy to avoid SettingWithCopyWarning
    data_local = data.copy()
    data_local['InvoiceDateOnly'] = data_local[timestamp].dt.date
    
    first_date = data_local.groupby([customer_id, 'InvoiceDateOnly']).first().reset_index()
    frequency_df = first_date.groupby([customer_id]).agg({order_id: "nunique"}).reset_index()
    frequency_df.rename(columns={order_id: 'Frequency'}, inplace=True)

    countOfPurchByDate = data_local.groupby([customer_id, 'InvoiceDateOnly']).size().reset_index(name='Count Of Purchases On Same Day')
    sumOfPurchByDate = data_local.groupby([customer_id, 'InvoiceDateOnly'])[monetary_value].sum().reset_index(name='Total Purchase Value On Same Day')

    # Merge count and sum instead of index assignment for safety
    combined_daily = pd.merge(countOfPurchByDate, sumOfPurchByDate, on=[customer_id, 'InvoiceDateOnly'])
    
    valueOfFirstPurchase = combined_daily.groupby(customer_id).first().reset_index()

    monetary_df = data_local.groupby(customer_id)[monetary_value].sum().reset_index(name='Total Monetary Value')
    monetary_df = monetary_df.merge(valueOfFirstPurchase[[customer_id, "Total Purchase Value On Same Day"]], on=customer_id, how='left')
    monetary_df = monetary_df.rename(columns={"Total Purchase Value On Same Day": 'First Time Purchase Value'})

    RFM = pd.merge(frequency_df, recency_df, on=customer_id)
    RFM = pd.merge(RFM, monetary_df, on=customer_id)
    RFM["Frequency Minus 1"] = RFM["Frequency"] - 1

    RFM["Average Time Between Orders In Days"] = RFM['Age In Days'] / RFM['Frequency Minus 1']
    RFM["Average Monetary Value Per Order"] = (RFM['Total Monetary Value'] - RFM['First Time Purchase Value']) / RFM['Frequency Minus 1']
    
    RFM = RFM[[customer_id, 'Frequency', 'Frequency Minus 1', 'Recency In Days', 'Age In Days',
               'Earliest Order Date', 'Most Recent Order Date', "Total Monetary Value",
               "First Time Purchase Value", "Average Time Between Orders In Days", "Average Monetary Value Per Order"]]

    RFM = RFM.sort_values('Frequency', ascending=False)

    RFM["Average Time Between Orders In Days"] = RFM["Average Time Between Orders In Days"].replace([np.inf, -np.inf], 0)
    RFM["Average Monetary Value Per Order"] = RFM["Average Monetary Value Per Order"].fillna(0)

    return RFM

if __name__ == "__main__":
    data = fit_models('Ecommerce.csv')
    
    aggregate_by_day = aggregate_by_time(data,'session_id','customer_id',
                                      'visit_date','revenue',granularity="D")

    aggregate_by_month = aggregate_by_time(data,'session_id','customer_id',
                                      'visit_date','revenue',granularity="ME")

    df_TotalPurchaseValueByMonth = aggregate_by_month[['visit_date','revenue']].groupby('visit_date').sum()
    
    ax = df_TotalPurchaseValueByMonth.plot(kind='line', rot='vertical', title='Revenue vs Time')

    # df_product_quantity = pd.DataFrame(data.groupby("product_category")["quantity"].sum())

    # df_product_quantity.sort_values("quantity",ascending=True).tail(15).plot(kind = 'barh')

    # df_customer_spend = pd.DataFrame(data.groupby("customer_id")["revenue"].sum())

    # df_customer_spend.sort_values("revenue",ascending=True).tail(15).plot(kind = 'barh')

    # df_customer_spend.hist()

    customer_earliest_order_date = pd.DataFrame(aggregate_by_day.groupby(['customer_id'])['visit_date'].min())

    customer_earliest_order_date.rename(columns={'visit_date':'Earliest Order Date'},inplace=True)
    customer_earliest_order_date.reset_index(inplace=True)
    
    customer_joining_by_month = pd.DataFrame(customer_earliest_order_date.groupby([
                                pd.Grouper(key='Earliest Order Date', freq='ME')
    ])["customer_id"].count())    
    # customer_joining_by_month.plot(title='No Of New Customers By Earliest Purchase Date By Month')
    
    min_date = min(data['visit_date'])
    max_date = max(data['visit_date'])
    full_max = max_date 

    # print(min_date, max_date)

    # plt.show()

    #Test-Train Split

    data = data.sort_values('visit_date')

    train_data, test_data = train_test_split(data,test_size=0.3,shuffle=False)

    train_min_date=min(train_data['visit_date'])
    train_max_date = max(train_data['visit_date'])

    print("Training Data Max Date: "+str(train_max_date))
    print("Training Data Min Date: "+str(train_min_date))
    print("Training Data Total Dur Days: " +str((train_max_date-train_min_date).days))
    print("Training Data Total Dur Weeks: " +str(round((train_max_date-train_min_date).days/7,1)))
    print("Training Data Total Dur Mths: " +str(round((train_max_date-train_min_date).days/30.417,1)))
    
    # print(train_data.describe())

    # Develop function to calculate RFM Characteristics

    RFM_train = create_RFM_Table(train_data,'session_id','customer_id','visit_date','revenue',"False")
    
    # print(RFM_train.head())   

    RFM = create_RFM_Table(data,'session_id','customer_id','visit_date','revenue',"False")
    print(RFM.head())

