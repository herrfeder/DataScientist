import pandas as pd
from IPython.core import debugger
debug = debugger.Pdb().set_trace
import pathlib
import os
import sys



class ErrorComparer():
    pass

class ChartData():
    
    def __init__(self, window_size=30):
    
        charts = ["bitcoin_hist", "sp500_hist", "dax_hist", "googl_hist", "gold_hist", "alibaba_hist", "amazon_hist"]
        sents = ["bitcoin_sent_df", "economy_sent_df"]
        
        self._chart_df = ""
        
        self.base_path = pathlib.Path(__file__).parent.resolve()
        self.data_path = os.path.join(self.base_path, "data")
        
        self.df_d = self.read_data_sources()
        for chart in charts:
            self.df_d[chart] = self.prep_charts(chart)   
        
        self.df_d["trend_df"] = self.prepare_trend("trend_df")

        for sent in sents:
            self.df_d[sent] = self.prepare_sent(sent)

        self.merge_dict_to_chart_df()
        
        
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
    

    def apply_boll_bands(self, df_string, price_col="Price",window_size=30, append_chart=False):
        try:
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
            return df
        
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
            df["price_norm"] = df["Price"] / df["Price"].max()

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

    def merge_dict_to_chart_df(self, chart_col="Price"):

        self.chart_df = self.df_d["bitcoin_hist"][[chart_col]]
        self.chart_df.columns = ["bitcoin_{}".format(x) for x in self.chart_df.columns]

        for stock in ["sp500_hist", "dax_hist", "googl_hist", "gold_hist", "alibaba_hist", "amazon_hist"]:
            stock_name = stock.split("_")[0]
            self.chart_df = self.chart_df.merge(self.df_d[stock][[chart_col]], 
                                                left_index=True, 
                                                right_index=True)
            self.chart_df = self.chart_df.rename(columns={chart_col:stock_name+"_Price"})

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

            
    def append_to_chart_df(self, append_df, prefix_name, right_key="Date"):
        
        append_df.columns = ["{}_{}".format(prefix_name, col) for col in append_df.columns]
        self.chart_df = self.chart_df.merge(append_df, left_index=True, right_index=True)
        

class ShiftChartData(ChartData):
    def __init__(self, fixed_cols):
        super().__init__()    
        
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
        if len(set(fixed_cols).intersection(self.chart_df.columns)) == len(fixed_cols):
            print("This Column doesn't exist in chart_df, using first column instead")
            return self.chart_df.columns[0]
        else:
            return fixed_cols
    
    def get_shift_cols(self):
        cols = list(self.chart_df.columns)
        for fix_col in self.fixed_cols:
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