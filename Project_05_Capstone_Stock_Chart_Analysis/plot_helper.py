import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
import pandas as pd
import numpy as np


def apply_layout(fig, title="", height=1250):
    
    fig.update_layout(height=height, title_text=title)
    layout = fig["layout"]
    layout["paper_bgcolor"] = "#002b36"   
    layout["plot_bgcolor"] = "#1f2630"
    layout["font"]["color"] = "#2cfec1"
    layout["title"]["font"]["color"] = "#2cfec1"
    layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    layout["xaxis"]["gridcolor"] = "#5b5b5b"
    layout["yaxis"]["gridcolor"] = "#5b5b5b"
    layout["margin"]["t"] = 75
    layout["margin"]["r"] = 0
    layout["margin"]["b"] = 0
    layout["margin"]["l"] = 0

    return fig



def get_gru_plot(df, fig="", title="", offset=31, dash=False):
    
    if not fig:
        fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
        
    
    df_mean = df.rolling(window=10,min_periods=1).mean()
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df[0],
                             line=dict(color='blue'),
                             name="GRU Prediction"), row=1, col=1)
    
    if dash:
        return apply_layout(fig, title)
    else:
        return fig
    


def get_ari_plot(df, fig="", title="", offset=31,conf_int=False, dash=False):
    
    if not fig:
        fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
        
    
    df_mean = df.predicted_mean.rolling(window=10,min_periods=1).mean()
    
    fig.add_trace(go.Scatter(x=df.predicted_mean.index + DateOffset(offset), 
                             y=df_mean,
                             line=dict(color='green'),
                             name="SARIMAX Prediction"), row=1, col=1)
    
    if conf_int:
        
        fig.add_trace(go.Scatter(x=df.predicted_mean.index + DateOffset(offset), 
                             y=df.conf_int()["lower bitcoin_Price"],
                             fill='tonexty',
                             fillcolor='rgba(166, 217, 193,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="SARIMAX Lower Confidence Interval"))


        fig.add_trace(go.Scatter(x=df.predicted_mean.index + DateOffset(offset), 
                             y=df.conf_int()["upper bitcoin_Price"],
                             fill='tonexty',
                             fillcolor='rgba(166, 217, 193,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="SARIMAX Higher Confidence Interval"))
        
    
    if dash:
        return apply_layout(fig, title)
    else:
        return fig
    

def price_plot(df, real_30=pd.DataFrame(),fig="", title="", boll=True, dash=False, names=["BTC Price",
                                           "BTC 30 Day Moving Average",
                                           "BTC Upper Bollinger Band",
                                           "BTC Lower Bollinger Band"]):
    if not fig:
        fig = make_subplots(
                        rows=1, 
                        cols=1, 
                        shared_xaxes=True)
     
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['bitcoin_Price'],
                             name=names[0]), row=1, col=1)
    
    if not real_30.empty:
        fig.add_trace(go.Scatter(x=real_30.index,
                                 y=real_30['bitcoin_Price'],
                                 name="Future Real Bitcoin Price",
                                 line=dict(color='grey', dash='dot')), row=1, col=1)
    if boll:
        fig.add_trace(go.Scatter(x=df.index, 
                                 y=df['bitcoin_30_day_ma'],
                                 name=names[1]), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, 
                                 y=df['bitcoin_boll_upp'],
                                 fill='tonexty',
                                 fillcolor='rgba(231,107,243,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name=names[2]), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, 
                                 y=df['bitcoin_boll_low'],
                                 fill='tonexty',
                                 fillcolor='rgba(231,50,243,0.2)',
                                 line=dict(color='rgba(255,255,255,0)'),
                                 name=names[3]), row=1, col=1) 
        
        
    if dash:
        return apply_layout(fig, title, height=600)
    else:
        return fig


def exploratory_plot(df, title="",dash=False):

    fig = make_subplots(
                        rows=4, 
                        cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08,
                        subplot_titles=("Bitcoin Historic Price Chart with Bollinger Bands (30-day)", 
                                        "Other Normalized Stock Price Historic Charts", 
                                        "Historic Google Trends", 
                                        "Historic Sentiments Twitter")
                        )
    
    fig = price_plot(df, fig=fig)

    y_2_list = ["sp500_Price_norm", 
                "dax_Price_norm",
                "googl_Price_norm",
                "gold_Price_norm",
                "amazon_Price_norm",
                "alibaba_Price_norm"]
    
    name_2_list = ["SP500 Normed Close",
                   "DAX Normed Close",
                   "GOOGLE Normed Close",
                   "GOLD Normed Close",
                   "AMAZON Normed Close",
                   "ALIBABA Normed Close"]
    
    for y, name in zip(y_2_list, name_2_list):
        fig.add_trace(go.Scatter(x=df.index, 
                                y=df[y],
                                name=name), row=2, col=1)

   
    fig.add_trace(go.Scatter(x=df.index,
                             y=df["bitcoin_Google_Trends"],
                             name="'Bitcoin' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["trading_Google_Trends"],
                             name="'Trading' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["cryptocurrency_Google_Trends"],
                             name="'Cryptocurrency' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["bitcoin_quot_sents"],
                             name="'Bitcoin' Sentiments"), row=4, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["economy_quot_sents"],
                             name="'#economy' Sentiments"), row=4, col=1)
    
    fig.update_yaxes(title_text="Absolute Price in $", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Price", row=2, col=1)
    fig.update_yaxes(title_text="Number of Query per day", row=3, col=1)
    fig.update_yaxes(title_text="Normalized Sentiment Quotient Value", row=4, col=1)

    if dash:
        return apply_layout(fig, title)
    else:
        fig.show()

        
def return_season_plot(df, column, title="", dash=False):
    
    series = pd.DataFrame(data=df[column].values, 
                          index=df.index, 
                          columns =[column]).dropna()

    result = seasonal_decompose(series.values, model='multiplicative', period=30)
    
    fig = make_subplots(
                        rows=3, 
                        cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.08,
                        )
    
    fig.add_trace(go.Scatter(x=df.index, 
                         y=result.trend,
                         name="Trend"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                         y=result.resid,
                         name="Residuals"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                             y=result.seasonal,
                             name="Seasonality"), row=3, col=1)
    
    fig.update_yaxes(title_text="Trend Of Chart", row=1, col=1)
    fig.update_yaxes(title_text="Residuals Of Chart", row=2, col=1)
    fig.update_yaxes(title_text="Seasonality Of Chart", row=3, col=1)
    
    
    if dash:
        return apply_layout(fig, title, height=800)
    else:
        fig.show()
        

def plot_val_heatmap(df, title="", height=1000, colormap="viridis", colorbar="Corr Coefficient",dash=False):
    
    coordinates = df.values.tolist()
    columns = list(df.columns)
    index = list(df.index)

    trace1 = {
            "type": "heatmap", 
            "x": columns, 
            "y": index, 
            "z": coordinates,
            "colorscale": colormap,
            "colorbar": {"title": "P-Value"}
            }

    data = trace1
    layout = {"title": title,
              "height": 1000}
    fig = go.Figure(dict(data=data, layout=layout))
    
    if dash:
        return apply_layout(fig, title, height=800)
    else:
        fig.show()
        
        
def return_shift_corr(do, fixed="bitcoin_Price", shift=-30, output="multi",dash=False):
    
    do.fixed_cols = fixed
    corr = do.single_shift(shift).corr()
    
    if output=="single":
        corr = corr[corr.index == fixed]
        return plot_val_heatmap(corr, height=200, dash=True)
    else:
        return plot_val_heatmap(corr, dash=True)
    

def return_granger_plot(df_path, title="", height=1000, colormap="viridis",dash=False):
    
    granger_df = pd.read_csv(df_path)
    granger_df.set_index("Unnamed: 0", inplace=True)
    granger_df[granger_df > 0.06] = np.nan
    return plot_val_heatmap(granger_df, 
                            title=title, 
                            height=height, 
                            colormap=colormap,
                            colorbar="P-Value",
                            dash=dash)

def return_cross_val_plot(split_dict, title="", height=1200, dash=False):
    
    params = ["CORR", "MSE", "RMSE", "R2"]
    
   
        
    valid_list = []
    forecast_list = []
    title_list = []
    for i in range(0,3):
        title_text = "Split {}<br>| ".format(i+1)
        for para in params:
            dict_val = split_dict.get("S_{}_{}".format(i,para),"") 
            if dict_val:
                title_text += para+" = "+str(np.round(dict_val,2))+" | "
        
        title_list.append(title_text)
        
        valid_plot = split_dict["S_{}_VALID".format(i)]
        
        if hasattr(valid_plot, "index"):
            valid_plot_index = valid_plot.index
        else:
            valid_plot_index = list(range(0, len(valid_plot)))
        
        valid_list.append(go.Scatter(x=valid_plot_index, 
                         y=valid_plot,
                         name="Real Bitcoin Price Split {}".format(i+1)))
        
        fore_plot = split_dict["S_{}_FORE".format(i)]
        
        if hasattr(fore_plot, "index"):
            fore_plot_index = valid_plot.index
        else:
            fore_plot_index = list(range(0, len(fore_plot)))
        
        forecast_list.append(go.Scatter(x=valid_plot_index, 
                         y=fore_plot,
                         name="Predicted Bitcoin Price Split {}".format(i+1)))
        
    fig = make_subplots(
                    rows=3, 
                    cols=1, 
                    vertical_spacing=0.08,
                    subplot_titles=(title_list[0], 
                                    title_list[1], 
                                    title_list[2]))
    
    index = 1
    for valid, forecast in zip(valid_list, forecast_list):
        fig.add_trace(valid, row=index, col=1)
        fig.add_trace(forecast, row=index, col=1)
        index = index + 1
    
  
    if dash:
        return apply_layout(fig, title, height=height)
    else:
        fig.show()
    
    return fig
