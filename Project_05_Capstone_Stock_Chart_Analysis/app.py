import json
import base64
import datetime
import requests
import pathlib
import math
import pandas as pd
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from dash.dependencies import Input, Output, State
from plotly import tools

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


# Dash App Layout
app.layout = html.Div(
    className="row",
    children=[
        # Interval component for live clock
        #dcc.Interval(id="interval", interval=1 * 1000, n_intervals=0),
        ## Interval component for ask bid updates
        #dcc.Interval(id="i_bis", interval=1 * 2000, n_intervals=0),
        # Interval component for graph updates
        #dcc.Interval(id="i_tris", interval=1 * 5000, n_intervals=0),
        # Interval component for graph updates
        #dcc.Interval(id="i_news", interval=1 * 60000, n_intervals=0),
        # Left Panel Div
        html.Div(
            className="three columns div-left-panel",
            children=[
                # Div for Left Panel App Info
                html.Div(
                    className="div-info",
                    children=[
                        html.Img(
                            className="logo", src=app.get_asset_url("dash-logo-new.png")
                        ),
                        html.H6(className="title-header", children="FOREX TRADER"),
                        html.P(
                            """
                            This app continually queries csv files and updates Ask and Bid prices
                            for major currency pairs as well as Stock Charts. You can also virtually
                            buy and sell stocks and see the profit updates.
                            """
                        ),
                    ],
                ),
                # Ask Bid Currency Div
                html.Div(
                    className="div-currency-toggles",
                    children=[
                        html.P(
                            id="live_clock",
                            className="three-col",
                            children=datetime.datetime.now().strftime("%H:%M:%S"),
                        ),
                        html.P(className="three-col", children="Bid"),
                        html.P(className="three-col", children="Ask"),
                        html.Div(
                            id="pairs",
                            className="div-bid-ask",
                            children=[
                                
                            ],
                        ),
                    ],
                ),
                # Div for News Headlines
                html.Div(
                    className="div-news",
                    children=[html.Div(id="news", children=[])],
                ),
            ],
        ),
        # Right Panel Div
        html.Div(
            className="nine columns div-right-panel",
            children=[
                # Top Bar Div - Displays Balance, Equity, ... , Open P/L
                html.Div(
                    id="top_bar", className="row div-top-bar", children=[]
                ),
                # Charts Div
                html.Div(
                    id="charts",
                    className="row",
                    children=[],
                ),
                # Panel for orders
                html.Div(
                    id="bottom_panel",
                    className="row div-bottom-panel",
                    children=[
                        html.Div(
                            className="display-inlineblock",
                            children=[
                                dcc.Dropdown(
                                    id="dropdown_positions",
                                    className="bottom-dropdown",
                                    options=[
                                        {"label": "Open Positions", "value": "open"},
                                        {
                                            "label": "Closed Positions",
                                            "value": "closed",
                                        },
                                    ],
                                    value="open",
                                    clearable=False,
                                    style={"border": "0px solid black"},
                                )
                            ],
                        ),
                        html.Div(
                            className="display-inlineblock float-right",
                            children=[
                                dcc.Dropdown(
                                    id="closable_orders",
                                    className="bottom-dropdown",
                                    placeholder="Close order",
                                )
                            ],
                        ),
                        html.Div(id="orders_table", className="row table-orders"),
                    ],
                ),
            ],
        ),
        # Hidden div that stores all clicked charts (EURUSD, USDCHF, etc.)
        html.Div(id="charts_clicked", style={"display": "none"}),
        # Hidden div for each pair that stores orders
        html.Div(
            children=[]),

        # Hidden Div that stores all orders
        #html.Div(id="orders", style={"display": "none"}),
    ],
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
    app.run_server(debug=True,  port=8050, host="0.0.0.0")
