import os
import pathlib
import re
import json
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
from matplotlib import cm
from matplotlib import colors
from scipy import interpolate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Initialize app

app = dash.Dash(
    __name__,
    url_base_pathname='/bitcoinprediction/',
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server

# Load data

APP_PATH = str(pathlib.Path(__file__).parent.resolve())




DEFAULT_COLORSCALE = [
"#3b3b3d",
"#8000ff",
"#6629fe",
"#4c50fc",
"#3079f7",
"#169bf2",
"#07bbea",
"#20d5e1",
"#3dead5",
"#56f7ca",
"#72febb",
"#8cfead",
"#a8f79c",
"#c2ea8c",
"#ded579",
"#f9bb66",
"#ff9b52",
"#ff793e",
"#ff5029",
"#ff2914",
"#ff0000",
]


DEFAULT_OPACITY = 0.5


# App layout


app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.Img(id="logo",width="100px", height="200px", src="https://www.nicepng.com/png/detail/21-216488_tis-but-a-scratch-butchers-tears.png"),
                html.H4(children="TIDE Hackathon Predicting Crisis in AFRICA"),
                html.P(
                    id="description",
                    children="This shows the monthly data for different metrics over all african countries",
                ),
            ],
        ),
        html.Div(
            id="app-container",
            children=[dbc.Row(),dbc.Row(),dbc.Row()]
                
        
        )
    ]
)
             

def exploratory_plot():
    fig = make_subplots(
    rows=4, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    subplot_titles=("Bitcoin Price Chart with Bollinger Bands (30-day)", 
                    "Other Normalized Stock Price Charts", 
                    "Google Trends", 
                    "Sentiments Twitter")
                    )

    #fig = go.Figure()

    fig.add_trace(go.Scatter(x=bitcoin_hist['Date'], 
                             y=bitcoin_hist['Price'],
                             name="BITCOIN Adjusted Close"), row=1, col=1)

    fig.add_trace(go.Scatter(x=bitcoin_hist['Date'], 
                             y=bitcoin_hist['30_day_ma'],
                             name="30 Day Moving Average"), row=1, col=1)

    fig.add_trace(go.Scatter(x=bitcoin_hist['Date'], 
                             y=bitcoin_hist['boll_upp'],
                             fill='tonexty',
                             fillcolor='rgba(231,107,243,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="Upper Bollinger Band"), row=1, col=1)

    fig.add_trace(go.Scatter(x=bitcoin_hist['Date'], 
                             y=bitcoin_hist['boll_low'],
                             fill='tonexty',
                             fillcolor='rgba(231,50,243,0.2)',
                             line=dict(color='rgba(255,255,255,0)'),
                             name="Lower Bollinger Band"), row=1, col=1)


    fig.add_trace(go.Scatter(x=sp500_hist['Date'], 
                             y=sp500_hist['price_norm'],
                             name="SP500 Normed Close"), row=2, col=1)

    fig.add_trace(go.Scatter(x=dax_hist['Date'], 
                             y=dax_hist['price_norm'],
                             name="DAX Normed Close"), row=2, col=1)


    fig.add_trace(go.Scatter(x=googl_hist['Date'], 
                             y=googl_hist['price_norm'],
                             name="GOOGLE Normed Close"), row=2, col=1)


    fig.add_trace(go.Scatter(x=trend_df.index,
                             y=trend_df["bitcoin"],
                             name="'Bitcoin' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=trend_df.index,
                             y=trend_df["trading"],
                             name="'Trading' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=trend_df.index,
                             y=trend_df["etf"],
                             name="'ETF' Google Trends"), row=3, col=1)

    fig.add_trace(go.Scatter(x=bitcoin_sent_df["date"],
                             y=bitcoin_sent_df["quot"],
                             name="'Bitcoin' Sentiments"), row=4, col=1)

    fig.add_trace(go.Scatter(x=economy_sent_df["date"],
                             y=economy_sent_df["quot"],
                             name="'#economy' Sentiments"), row=4, col=1)


    fig.update_layout(height=1000, width=1500, title_text="Bitcoin Price Chart against different Indicators")

    fig.show()
                
                
             
if __name__ == "__main__":
    app.run_server(debug=True,  port=8000, host="0.0.0.0")
