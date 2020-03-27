import pandas as pd
from IPython.core import debugger
debug = debugger.Pdb().set_trace



class ErrorComparer():
    pass

class ChartData():
    
    def __init__(self):
    
        charts = ["bitcoin_hist", "sp500_hist", "dax_hist", "googl_hist"]
        sents = ["bitcoin_sent_df", "economy_sent_df"]
        
        self.df_d = self.read_data_sources()
        for chart in charts:
            self.df_d[chart] = self.prep_charts(self.df_d[chart])   

        self.df_d["bitcoin_hist"] = self.apply_bollinger_bands(self.df_d["bitcoin_hist"])
            
        self.df_d["trend_df"]  = self.prepare_trend(self.df_d["trend_df"])

        for sent in sents:
            self.df_d[sent] = self.prepare_sent(self.df_d[sent])


        self.chart_df = self.merge_dict_to_df(self.df_d)
        
    
    def get_chart_df(self):
        return self.chart_df
        
    def read_data_sources(self):
        data_source_d = {}
        data_source_d["bitcoin_hist"] = pd.read_csv("data/Bitcoin Historical Data - Investing.com.csv")
        data_source_d["sp500_hist"] = pd.read_csv("data/S&P 500 Historical Data.csv")
        data_source_d["dax_hist"] = dax_hist = pd.read_csv("data/DAX Historical Data.csv")
        data_source_d["googl_hist"] = pd.read_csv("data/GOOGL Historical Data.csv")
        data_source_d["trend_df"] = pd.read_csv("data/trends_bitcoin_cc_eth_trading_etf.csv")
        data_source_d["bitcoin_sent_df"] = pd.read_csv("data/bitcoin_sentiments.csv")
        data_source_d["economy_sent_df"] = pd.read_csv("data/economy_sentiments.csv")
        data_source_d["gold_hist"] = pd.read_csv("data/GOLD Historical Data.csv")

        return data_source_d
    


    def prep_charts(self, df, norm=False):
        df["Price"] = df.apply(convert_values, args=("Price",), axis=1)
        df["Open"] = df.apply(convert_values, args=("Open",), axis=1)
        df["High"] = df.apply(convert_values, args=("High",), axis=1)
        df["Low"] = df.apply(convert_values, args=("Low",), axis=1)
        df["Vol."] = df.apply(convert_vol, args=("Vol.",), axis=1)


        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by="Date").reset_index()
        del df["index"]

        if norm:
            df["price_norm"] = df["Price"] / df["Price"].max()

        return df

    def apply_bollinger_bands(self, df, window_size=30):
        df["30_day_ma"] = df["Price"].rolling(window_size, min_periods=1).mean()
        df["30_day_std"] = df["Price"].rolling(window_size, min_periods=1).std()
        df["boll_upp"] = df['30_day_ma'] + (df['30_day_std'] * 2)
        df["boll_low"] = df['30_day_ma'] - (df['30_day_std'] * 2)

        return df

    def prepare_trend(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")
        df = df.resample("D").sum()
        df.reset_index(inplace=True)

        return df

    def prepare_sent(self, df):
        df["date"] = pd.to_datetime(df["date"])
        df["quot"] = df["pos"] / df["neg"]

        return df

    def merge_dict_to_df(self, df_d):

        corr_df = df_d["bitcoin_hist"][["Date","Price", "High", "Low", "30_day_ma", "30_day_std", "boll_upp", "boll_low"]]
        corr_df.columns = ["bitcoin_{}".format(x) for x in corr_df.columns]

        for stock in ["sp500_hist", "dax_hist", "googl_hist"]:
            stock_name = stock.split("_")[0]
            corr_df = corr_df.merge(df_d[stock][["Date","Price"]], left_on="bitcoin_Date", right_on="Date", how="right")
            corr_df = corr_df.rename(columns={"Price":stock_name+"_Price"})
            corr_df.drop(columns=["Date"], inplace=True)

        corr_df = corr_df.merge(df_d["trend_df"], 
                                left_on="bitcoin_Date", 
                                right_on="date").drop(columns=["date", "etf", "ethereum", "isPartial"])

        corr_df = corr_df.rename(columns={"bitcoin":"bitcoin_Google_Trends", 
                                          "cryptocurrency":"cryptocurrency_Google_Trends",
                                          "trading":"trading_Google_Trends"})

        corr_df = corr_df.merge(df_d["bitcoin_sent_df"], left_on="bitcoin_Date", right_on="date").drop(columns=["date", "length"])
        corr_df = corr_df.rename(columns={"pos": "bitcoin_pos_sents",
                                          "neg": "bitcoin_neg_sents",
                                          "quot": "bitcoin_quot_sents"})

        corr_df = corr_df.merge(df_d["economy_sent_df"], left_on="bitcoin_Date", right_on="date").drop(columns=["date", "length"])
        corr_df = corr_df.rename(columns={"pos": "economy_pos_sents",
                                          "neg": "economy_neg_sents",
                                          "quot": "economy_quot_sents"})

        return corr_df


    
    
    
def convert_values(row, col):
    val = row[col].replace(",","")
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