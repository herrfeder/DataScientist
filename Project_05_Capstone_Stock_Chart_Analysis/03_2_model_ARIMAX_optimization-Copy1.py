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

import json

do = data_prep_helper.ValidateChartData(chart_col=["Price", "High", "Low"])


ext_cols = []
for col in train.columns:
    if (col.endswith('month')) or (col.endswith('week')) or (col.startswith('month')):
        ext_cols.append(col)
        
result_dict = {}

for i in range(3,12):
    result_list = []
    for comb in itertools.combinations(ext_cols, i):
        result_dict = {}
        for index, train, test in enumerate(do.gen_return_splits()):

            exog_s1i1 = train[list(comb)]
            exog = valid[list(comb)]
            s1i1 = train['bitcoin_Price']
            arimax = sm.tsa.statespace.SARIMAX(s1i1, exog=exog_s1i1,
                                       enforce_invertibility=False, 
                                       enforce_stationarity=False, 
                                       freq='D', iprint=-1).fit()

            forecast = arimax.get_forecast(steps=len(valid), exog=exog)

            result_dict["split_{}_CORR".format(index)] = np.corrcoef(forecast.predicted_mean,valid["bitcoin_Price"].values)[0][1]
            result_dict["S_{}_RMSE".format(index)] = sqrt(mean_squared_error(forecast.predicted_mean, valid["bitcoin_Price"]))
        result_dict["FEATURES"] = str(comb)
        result_list.append(result_dict)
    
    result_df = pd.DataFrame(result_list)
    result_df["NUM_FEATURES"] = i   
    result_df.to_csv("arimax_results/arimax_split_combinations_results_{}.csv".format(i))
    