import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
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
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['bitcoin_Price'],
                             name="BTC Adjusted Close"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['bitcoin_30_day_ma'],
                             name="BTC 30 Day Moving Average"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['bitcoin_boll_upp'],
                             fill='tonexty',
                             fillcolor='rgba(231,107,243,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="BTC Upper Bollinger Band"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['bitcoin_boll_low'],
                             fill='tonexty',
                             fillcolor='rgba(231,50,243,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="BTC Lower Bollinger Band"), row=1, col=1)

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
    

