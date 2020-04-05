import json
import base64
import datetime
import requests
import pathlib
import math
import pandas as pd
import flask
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import data_prep_helper

from dash.dependencies import Input, Output, State
from plotly import tools

do = data_prep_helper.ValidateChartData(chart_col=["Price", "High", "Low", "Price_norm"])
do.apply_boll_bands("bitcoin_hist", append_chart=True)


acc_str = ["View Data", 
           "Correlation Analysis",
           "Causality Analysis"]
    
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


def make_item(i):
    # we use this function to make the example items to avoid code duplication
    return dbc.Card(
        [
            dbc.CardHeader(
                    dbc.Button(
                        f"{i}",
                        color="link",
                        id=f"group-{i}-toggle",
                    
                    )
                
            ),
            dbc.Collapse(
                dbc.CardBody(f"This is the content of group {i}..."),
                id=f"collapse-{i}",
            ),
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

    df = do.chart_df
    
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


    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['sp500_Price_norm'],
                             name="SP500 Normed Close"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['dax_Price_norm'],
                             name="DAX Normed Close"), row=2, col=1)


    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['googl_Price_norm'],
                             name="GOOGLE Normed Close"), row=2, col=1)


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

    fig.update_layout(height=1000, title_text="Bitcoin Price Chart against different Indicators")

    layout = fig["layout"]
    layout["paper_bgcolor"] = "#1f2630"
    layout["plot_bgcolor"] = "#1f2630"
    layout["font"]["color"] = "#2cfec1"
    layout["title"]["font"]["color"] = "#2cfec1"
    layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    layout["xaxis"]["gridcolor"] = "#5b5b5b"
    layout["yaxis"]["gridcolor"] = "#5b5b5b"
    layout["margin"]["t"] = 75
    layout["margin"]["r"] = 50
    layout["margin"]["b"] = 100
    layout["margin"]["l"] = 50
    
    return fig


# NAVBAR always on top of website
NAVBAR = dbc.Navbar(
    children=[
       
            dbc.Row(
                [
                    dbc.Col(html.A(html.Img(src="https://upload.wikimedia.org/wikipedia/commons/3/3b/Udacity_logo.png", height="20px"), href="https://www.udacity.com")),
                    dbc.Col(dbc.NavbarBrand("Recommendations with IBM Project"), className="ml-2"),
                    dbc.Col(dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("LinkedIn",
                                        href="https://www.linkedin.com/in/davidlassig/"),
                            dbc.DropdownMenuItem("Github Repo", 
                                        href="https://github.com/herrfeder/Udacity-Project-Recommendations-With-IBM-Webapp.git"),
                          
                        ],
                        nav=False,
                        in_navbar=True,
                        label="by David Lassig",
                        style={"color": "white", "font-size": 10, "font-weight":"lighter"},
                    )),
                    

                ],
                align="center",
                no_gutters=True,
            ),
           
        
    ],
    color="dark",
    dark=True,
    sticky="top",
)


LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="User Selection", className="display-5"),
        html.Hr(className="my-2"),
        html.Div([
                    make_item("View Data"), 
                    make_item("Correlation Analysis"), 
                    make_item("Causality Analysis")], className="accordion"),
              
    ]
)

CHART_PLOT = [dbc.CardHeader(html.H5("test")),
              dbc.CardBody(dcc.Loading(dcc.Graph(id="chart_plot",
                       figure=exploratory_plot())))
             ]

TABLE_VIEW = [dbc.CardHeader(html.H5("10 Most Similiar Users")),
              dbc.CardBody(dcc.Loading(dash_table.DataTable(id="usertableview")))
                ]

BODY = dbc.Container([
            dbc.Row(
                [
                    dbc.Col(LEFT_COLUMN, md=3),
                    dbc.Col(CHART_PLOT, md=9),
                ],
                style={"marginTop": 30},
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(""), width=8),
                    dbc.Col(dbc.Card(""), width=4),
                    
                ]
            )], fluid=True)


### Init webapp ###


app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.SOLAR], 
                url_base_pathname="/bitcoinprediction/",
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
                ],
               )

app.layout = html.Div(children=[NAVBAR, BODY])
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

server = app.server

           



@app.callback(
    Output("chart_plot", "figure"),
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str],
    [State("chart_plot", "figure")]
)    
def show_plot(acc_01, acc_02, acc_03, figure):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str[0] in button_id):
        print("blah")
        return exploratory_plot()
    else:
        return exploratory_plot()
    
@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in acc_str],
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str],
    [State(f"collapse-{i}", "is_open") for i in acc_str],
)
def toggle_accordion(n1, n2, n3, is_open1, is_open2, is_open3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str[0] in button_id) and n1:
        return not is_open1, False, False
    elif (acc_str[1] in button_id) and n2:
        return False, not is_open2, False
    elif (acc_str[2] in button_id) and n3:
        return False, False, not is_open3
    return False, False, False        

             
if __name__ == "__main__":
    app.run_server(debug=True,  port=8050, host="0.0.0.0")
