import data_prep_helper
import statsmodels.api as sm
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool
import os


do = data_prep_helper.ValidateChartData(chart_col=["Price", "High", "Low"])

df = data_prep_helper.ShiftChartData.get_causal_const_shift(do.chart_df,)


ext_cols = []
for col in df.columns:
    if (col.endswith('month')) or (col.endswith('week')) or (col.startswith('month')):
        ext_cols.append(col)
        
def feature_comb_iter(i):
    result_list = []
    for comb in itertools.combinations(ext_cols, i):
        result_dict = {}
        split_index = 0
        for train, test in do.gen_return_splits():
            split_index = split_index + 1
            exog_s1i1 = train[list(comb)]
            exog = test[list(comb)]
            s1i1 = train['bitcoin_Price']
            arimax = sm.tsa.statespace.SARIMAX(s1i1, exog=exog_s1i1,
                                       enforce_invertibility=False, 
                                       enforce_stationarity=False, 
                                       freq='D').fit(disp=0)

            forecast = arimax.get_forecast(steps=len(test), exog=exog)

            result_dict["split_{}_CORR".format(split_index)] = np.corrcoef(forecast.predicted_mean,test["bitcoin_Price"].values)[0][1]
            result_dict["S_{}_RMSE".format(split_index)] = sqrt(mean_squared_error(forecast.predicted_mean, test["bitcoin_Price"]))
    
        result_dict["FEATURES"] = str(comb)
        result_list.append(result_dict)
        
        print(str(comb))
        print(result_dict)
        print()
    
    result_df = pd.DataFrame(result_list)
    result_df["NUM_FEATURES"] = i   
    result_df.to_csv("arimax_results/arimax_split_combinations_results_{}.csv".format(i))
    
    
if __name__ == '__main__':
    pool = Pool(os.cpu_count())           # Create a multiprocessing Pool
    pool.map(feature_comb_iter, range(3,12))  # process range