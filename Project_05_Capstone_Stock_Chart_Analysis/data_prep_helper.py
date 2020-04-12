import pandas as pd
from IPython.core import debugger
debug = debugger.Pdb().set_trace
import pathlib
import os
import sys
import numpy as np
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error



class ErrorComparer():
    pass

class ChartData():
    
    def __init__(self, window_size=30, chart_col="Price"):
    
        charts = ["bitcoin_hist", "sp500_hist", "dax_hist", "googl_hist", "gold_hist", "alibaba_hist", "amazon_hist"]
        sents = ["bitcoin_sent_df", "economy_sent_df"]
        
        self._chart_df = ""
        
        self.base_path = pathlib.Path(__file__).parent.resolve()
        self.data_path = os.path.join(self.base_path, "data")
        
        self.df_d = self.read_data_sources()
        for chart in charts:
            self.df_d[chart] = self.prep_charts(chart, norm=True)   
        
        self.df_d["trend_df"] = self.prepare_trend("trend_df")

        for sent in sents:
            self.df_d[sent] = self.prepare_sent(sent)

        self.merge_dict_to_chart_df(chart_col)
        
        
    @property
    def chart_df(self):
        return self._chart_df
    
    @chart_df.setter
    def chart_df(self, df):
        self._chart_df = df
        
    @property
    def df_d(self):
        return self._df_d
    
    @df_d.setter
    def df_d(self, df_dict):
        self._df_d = df_dict
        
        
    def read_data_sources(self):
        data_source_d = {}
        data_source_d["bitcoin_hist"] = pd.read_csv(
            os.path.join(self.data_path, "Bitcoin Historical Data - Investing.com.csv"))
        data_source_d["sp500_hist"] = pd.read_csv(
            os.path.join(self.data_path, "S&P 500 Historical Data.csv"))
        data_source_d["dax_hist"] = dax_hist = pd.read_csv(
            os.path.join(self.data_path, "DAX Historical Data.csv"))
        data_source_d["googl_hist"] = pd.read_csv(
            os.path.join(self.data_path,"GOOGL Historical Data.csv"))
        data_source_d["trend_df"] = pd.read_csv(
            os.path.join(self.data_path, "trends_bitcoin_cc_eth_trading_etf.csv"))
        data_source_d["bitcoin_sent_df"] = pd.read_csv(
            os.path.join(self.data_path, "bitcoin_sentiments.csv"))
        data_source_d["economy_sent_df"] = pd.read_csv(
            os.path.join(self.data_path, "economy_sentiments.csv"))
        data_source_d["gold_hist"] = pd.read_csv(
            os.path.join(self.data_path, "GOLD Historical Data.csv"))
        data_source_d["alibaba_hist"] = pd.read_csv(
            os.path.join(self.data_path, "BABA Historical Data.csv"))
        data_source_d["amazon_hist"] = pd.read_csv(
            os.path.join(self.data_path, "AMZN Historical Data.csv"))
        return data_source_d
    

    def apply_boll_bands(self, 
                         df_string="", 
                         price_col="Price", 
                         window_size=30, 
                         append_chart=False, 
                         ext_df=pd.DataFrame()):
        try:
            if not ext_df.empty:
                df = ext_df
            else:
                df = self.df_d[df_string]
        except:
            print("Not found this DataFrame name")
        
        prefix = df_string.split("_")[0]
        
        df["30_day_ma"] = df[price_col].rolling(window_size, min_periods=1).mean()
        df["30_day_std"] = df[price_col].rolling(window_size, min_periods=1).std()
        df["boll_upp"] = df['30_day_ma'] + (df['30_day_std'] * 2)
        df["boll_low"] = df['30_day_ma'] - (df['30_day_std'] * 2)
        
        if append_chart:
            self.append_to_chart_df(df[["30_day_ma", "30_day_std", "boll_upp", "boll_low"]], prefix)
        else:
            if not ext_df.empty:
                df.columns = ["{}_{}".format(prefix, col) if not col.startswith(prefix) else col for col in df.columns]
                return df
            else:
                return self.append_to_chart_df(df[["30_day_ma", "30_day_std", "boll_upp", "boll_low"]], prefix, inplace=False)
        
    def prep_charts(self, chart_df_str, norm=False):
        try:
            df = self.df_d[chart_df_str]
        except:
            print("Not found this DataFrame name")
        
        df["Price"] = df.apply(convert_values, args=("Price",), axis=1)
        df["Open"] = df.apply(convert_values, args=("Open",), axis=1)
        df["High"] = df.apply(convert_values, args=("High",), axis=1)
        df["Low"] = df.apply(convert_values, args=("Low",), axis=1)
        df["Vol."] = df.apply(convert_vol, args=("Vol.",), axis=1)


        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date").reset_index()
        del df["index"]
        df = df.set_index("Date")

        if norm:
            df["Price_norm"] = df["Price"] / df["Price"].max()

        return df

    
    def prepare_trend(self, trend_df_str):
        try:
            df = self.df_d[trend_df_str]
        except:
            print("Not found this DataFrame name")
            return
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")
        df.index.name = "Date"
        df = df.resample("D").sum()
        
        return df
        
    def prepare_sent(self, sent_df_str):
        try:
            df = self.df_d[sent_df_str]
        except:
            print("Not found this DataFrame name")
            return
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index.name = "Date"
        df["quot"] = df["pos"] / df["neg"]
        
        return df

    def merge_dict_to_chart_df(self, chart_col=["Price"]):
        
        if not isinstance(chart_col, list):
            chart_col = [chart_col]
            
        self.chart_df = self.df_d["bitcoin_hist"][chart_col]
        self.chart_df.columns = ["bitcoin_{}".format(x) for x in self.chart_df.columns]

        for stock in ["sp500_hist", "dax_hist", "googl_hist", "gold_hist", "alibaba_hist", "amazon_hist"]:
            stock_name = stock.split("_")[0]
            self.chart_df = self.chart_df.merge(self.df_d[stock][chart_col], 
                                                left_index=True, 
                                                right_index=True)
            self.chart_df = self.chart_df.rename(columns={col:"{}_{}".format(stock_name, col) for col in chart_col})

        self.chart_df = self.chart_df.merge(self.df_d["trend_df"], 
                                            left_index=True, 
                                            right_index=True).drop(columns=["etf", "ethereum", "isPartial"])
        self.chart_df = self.chart_df.rename(columns={"bitcoin":"bitcoin_Google_Trends", 
                                                      "cryptocurrency":"cryptocurrency_Google_Trends",
                                                      "trading":"trading_Google_Trends"})

        for sent in ["bitcoin_sent_df", "economy_sent_df"]:
            sent_name = sent.split("_")[0]
            self.chart_df = self.chart_df.merge(self.df_d[sent], 
                                                left_index=True, 
                                                right_index=True).drop(columns=["length"])
            self.chart_df = self.chart_df.rename(columns={"pos": sent_name+"_pos_sents",
                                              "neg": sent_name+"_neg_sents",
                                              "quot": sent_name+"_quot_sents"})
            
            
        self.chart_df = self.chart_df.resample('D').interpolate()

            
    def append_to_chart_df(self, append_df, prefix_name, right_key="Date", inplace=False):
        
        append_df.columns = ["{}_{}".format(prefix_name, col) for col in append_df.columns]
        if not inplace:
            return self.chart_df.merge(append_df, left_index=True, right_index=True)
        else:
            self.chart_df = self.chart_df.merge(append_df, left_index=True, right_index=True)
            
            
    def get_growth(self, day, past, cols=["bitcoin_Price"]):
        
        past_days = past
        window = int(np.round(abs(past_days)/2))
        
        past_days_df = self.chart_df[self.chart_df.index < day].iloc[past_days:,:].rolling(window=window, min_periods=1 ).mean()
        
        if not isinstance(cols,list):
                cols = [cols]
        
        growth_dict = {}
        for col in cols:
            growth_dict[col] = np.round(100 - ((past_days_df[col][0]/past_days_df[col][-1])*100),2)
            
        return growth_dict
        

class ShiftChartData(ChartData):
    def __init__(self, fixed_cols="bitcoin_Price", window_size=30, chart_col="Price"):
        super().__init__(window_size, chart_col)    
        
        self._fixed_cols = fixed_cols
    
    @property
    def fixed_cols(self):
        return self._fixed_cols
    
    @fixed_cols.setter
    def fixed_cols(self, fixed_cols):
        if not isinstance(fixed_cols, list):
            fixed_cols = [fixed_cols]
            
        self._fixed_cols = self.check_fixed_cols(fixed_cols)
    
    
    def check_fixed_cols(self, fixed_cols):
        if not len(set(fixed_cols).intersection(set(self.chart_df.columns))) == len(fixed_cols):
            print("This Column {} doesn't exist in chart_df, using first column instead".format(fixed_cols))
            return [self.chart_df.columns[0]]
        else:
            return fixed_cols
    
    def get_shift_cols(self):
        cols = list(self.chart_df.columns)
        for fix_col in self._fixed_cols:
            print(fix_col)
            cols.remove(fix_col)
        
        return cols
        
    def single_shift(self, shift_val=-1):
        cols = self.get_shift_cols()
            
        df = self.chart_df[self.fixed_cols]
        for col in cols:
            df[col+"_"+str(shift_val)] = self.chart_df[col].shift(shift_val)
        
        return df
    
    def gen_multi_shift(self, shift_arr=[ 2, 1, 0, -1, -2]):
        cols = self.get_shift_cols()
        
        df = self.chart_df[self.fixed_cols]
        for shift_val in shift_arr:
            for col in cols:
                df[col] = self.chart_df[col].shift(shift_val)
            yield shift_val, df
            
    @staticmethod
    def get_most_causal_cols(df, past=""):
        
        opt_cols = ['bitcoin_Price', 'bitcoin_High', 'bitcoin_Google_Trends_prev_month',
                    'bitcoin_Google_Trends_prev_week', 'alibaba_High_prev_week',
                    'alibaba_Price_prev_week', 'bitcoin_Low_prev_month',
                    'bitcoin_Low_prev_week', 'bitcoin_High_prev_month',
                    'bitcoin_High_prev_week', 'cryptocurrency_Google_Trends_prev_week',
                    'bitcoin_Price_prev_month', 'bitcoin_Price_prev_week','cryptocurrency_Google_Trends',
                    'bitcoin_Low', 'alibaba_Price',
                    'alibaba_High', 'alibaba_Low',
                    'cryptocurrency_Google_Trends_prev_month', 'bitcoin_Google_Trends',
                    'alibaba_Low_prev_week', 'amazon_Price', 'month-1', 'month-2',
                    'alibaba_Low_prev_month', 'amazon_High',
                    'alibaba_Price_prev_month', 'alibaba_High_prev_month',
                    'amazon_High_prev_month', 'amazon_Low_prev_week',
                    'amazon_Price_prev_week', 'amazon_High_prev_week', 'sp500_High',
                    'amazon_Low', 'googl_Price', 'economy_pos_sents_prev_week', 'economy_pos_sents_prev_month']
        
        arimax_opt_cols = [
                   'bitcoin_Price_prev_week',
                   'bitcoin_Price_prev_month',
                   'alibaba_Price_prev_week',
                   'googl_Price_prev_month',
                   'bitcoin_trends_prev_week',
                   'bitcoin_trends_prev_month',
                   'cryptocurrency_trends_prev_week',
                   'cryptocurrency_trends_prev_month',
                   'month-1', 'month-2',
                   ]
        
        
        cols =   ['bitcoin_Price',
                 'bitcoin_High',
                 'bitcoin_Low',
                 'alibaba_Price',
                 'alibaba_High',
                 'alibaba_Low',
                 'amazon_Price',
                 'amazon_High',
                 'amazon_Low',
                 'googl_Price',
                 'googl_High',
                 'googl_Low',
                 'bitcoin_Google_Trends',
                 'cryptocurrency_Google_Trends',
                 'economy_pos_sents']
        
        week_cols = ["{}_prev_week".format(col) for col in cols]
        month_cols = ["{}_prev_month".format(col) for col in cols]
        
        if past=="now":
            return df[cols]
        elif past=="week":
            week_cols.extend(["month-1","month-2"])
            return df[week_cols]
        elif past=="month":
            month_cols.extend(["month-1","month-2"])
            return df[month_cols]
        elif past=="ari":
            return df[arimax_opt_cols]
        elif past=="all":
            week_cols.extend(month_cols)
            week_cols.extend(cols)
            all_cols = week_cols
            all_cols.extend(["month-1","month-2"])
            return df[all_cols]
        
        else:
            return df[opt_cols]
    
    @staticmethod
    def get_dummy_months(df):
        months = df.index.month
        dummy_months = pd.get_dummies(months)
        dummy_months.columns = ['month-%s' % m for m in range(1,len(dummy_months.columns)+1)]
        dummy_months.index = df.index
        
        df = pd.concat([df, dummy_months.iloc[:,:3]], axis=1)
        
        return df
    
    @staticmethod
    def get_causal_const_shift(df, past="", zeros="cut"):
        
        df = ShiftChartData.get_dummy_months(df)
        
        causal_cols = ["bitcoin_Price", 
                       "bitcoin_High",
                       "bitcoin_Low",
                       "alibaba_Price",
                       "alibaba_High",
                       "alibaba_Low",
                       "amazon_Price",
                       "amazon_High",
                       "amazon_Low",
                       "googl_Price",
                       "googl_High",
                       "googl_Low",
                       "bitcoin_Google_Trends",
                       "cryptocurrency_Google_Trends",
                       "economy_pos_sents"]
        try:
            for col in causal_cols:
                if past=="week":
                    df[col+"_prev_week"] = df[col].shift(8)
                elif past=="month":
                    df[col+"_prev_month"] = df[col].shift(31)
                else:
                    df[col+"_prev_week"] = df[col].shift(8)
                    df[col+"_prev_month"] = df[col].shift(31)
        except:
            pass
        
        if zeros=="cut" and past=="week":
            df = df.iloc[8:,:]
        elif zeros=="cut" and past=="month":
            df = df.iloc[31:,:]
        elif zeros=="zero":
            df.fillna(0, inplace=True)
        else:
            df = df.iloc[31:, :]
        
        return ShiftChartData.get_most_causal_cols(df, past)
    
    def return_train_test(self, split_factor=1500, past="all", zeros="cut"):
        if split_factor < 1:
            train = self.chart_df[:int(split_factor*(len(self.chart_df)))]
            test = self.chart_df[int(split_factor*(len(self.chart_df))):]
        else: 
            train = self.chart_df[:split_factor]
            test = self.chart_df[split_factor:]
        
        train = ShiftChartData.get_causal_const_shift(train, past=past, zeros=zeros)
        test = ShiftChartData.get_causal_const_shift(test, past=past, zeros=zeros)
        
        return train, test
    
    
    def gen_return_splits(self, splits=3, split_size=300, data_len=1800, past="all"):
        start_split = int(np.round((data_len/2)/split_size))
        end_split = start_split + splits
        
        
        for i in range(start_split,end_split):
            train = self.chart_df[:i*split_size]
            test = self.chart_df[i*split_size:(i+1)*split_size]
            
            train = ShiftChartData.get_causal_const_shift(train, past=past, zeros="cut")
            test = ShiftChartData.get_causal_const_shift(test, past=past, zeros="cut")
            
            yield train, test

            
class ModelData(ShiftChartData):
    def __init__(self, 
                 fixed_cols="bitcoin_Price", 
                 window_size=30, 
                 chart_col="Price", 
                 model_path="models",
                 opt_ari_feat=['bitcoin_Google_Trends_prev_month',
                               'cryptocurrency_Google_Trends_prev_month',
                               'alibaba_High_prev_month',
                               'amazon_High_prev_month',
                               'economy_pos_sents_prev_month']):
        super().__init__(fixed_cols, window_size, chart_col)    
               
        self.arimax_path = str(self.base_path.joinpath("models/sarimax_5_feat_month{}.pkl"))
        self.arimax_model = self.arimax_path.format("")
        self.arimax_split_model = [self.arimax_path.format("_S"+str(i)) for i in range(1,4)]
        self.arimax = pickle.load( open( self.arimax_model, "rb" ) )
        self.opt_ari_feat = opt_ari_feat

        self.train, self.test =  self.return_train_test()
        self.fore_exp, self.real_price = self.prep_ari_forecast()
        
        
    def get_forecast_dates(self):
        return list(self.fore_exp.index.strftime("%Y-%m-%d"))
        
    def prep_ari_forecast(self):
        feat_prep = [x.split("_prev_month")[0] for x in self.opt_ari_feat]
        feat_prep.append("bitcoin_Price")
        forecast_exp = self.chart_df[(self.chart_df.index <= self.test.index.max()) & (self.chart_df.index > self.train.index.max())][feat_prep]
        real_price = forecast_exp["bitcoin_Price"]
        forecast_exp.drop(columns = ["bitcoin_Price"], inplace=True)
        
        return forecast_exp, real_price     
        
    
    def ari_forecast(self, curr_day):
        curr_fore_exp = self.fore_exp[self.fore_exp.index <= curr_day]
        #try:
        #    curr_fore_exp["economy_pos_sents"] = curr_fore_exp["economy_pos_sents"].rolling(window=10,min_periods=1).mean()
        #except:
        #    pass
    
        curr_real_price = pd.DataFrame(self.real_price[self.real_price.index <= curr_day])
        curr_real_price = self.apply_boll_bands(df_string="bitcoin_Price", 
                                                price_col="bitcoin_Price",
                                                ext_df=curr_real_price)
        forecast = self.arimax.get_forecast(steps=len(curr_fore_exp), exog=curr_fore_exp)
        
        return forecast, curr_real_price
        
    def cross_validate_arimax(self):
            result_dict = {}
            split_index = 0
            for train, test in self.gen_return_splits():
        
                exog = test[self.opt_ari_feat]
            
                arimax = pickle.load( open( self.arimax_split_model[split_index], "rb" ) )
                forecast = arimax.get_forecast(steps=len(test), exog=exog)
        
                result_dict["S_{}_CORR".format(split_index)] = np.corrcoef(forecast.predicted_mean,test["bitcoin_Price"].values)[0][1]
                result_dict["S_{}_RMSE".format(split_index)] = sqrt(mean_squared_error(forecast.predicted_mean, test["bitcoin_Price"]))
        
                result_dict["S_{}_VALID".format(split_index)] = test["bitcoin_Price"]
                result_dict["S_{}_FORE".format(split_index)] = forecast.predicted_mean
            
                split_index = split_index + 1
                
            return result_dict

### APPLY FUNCTIONS ###
    
def convert_values(row, col):
    try:
        val = row[col].replace(",","")
    except:
        val = row[col]
    return float(val)

def convert_vol(row, col):
    
    letter = row[col][-1]
    val = row[col].rstrip(letter)
    if val == "":
        val = 0
    val = float(val)
    if letter=="M":
        val = val*1000000
    if letter=="K":
        val = val*1000
    
    return val    