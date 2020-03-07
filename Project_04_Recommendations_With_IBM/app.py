# -*- coding: utf-8 -*-
"""
Module doc string
"""
import pathlib
import re
import json
from datetime import datetime
import flask
import dash
import dash_table
import matplotlib.colors as mcolors
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from precomputing import add_stopwords
from dash.dependencies import Output, Input, State
from dateutil import relativedelta
from plotlywordcloud import plotly_wordcloud
from recommendengine import Recommender


DATA_PATH = pathlib.Path(__file__).parent.resolve()
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
USER_INTERACTIONS_CSV = "data/user-item-interactions.csv"
USER_ITEM_MATRIX_P = "data/user_item_matrix.p"
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
USER_INTERACTIONS_DF = pd.read_csv(DATA_PATH.joinpath(USER_INTERACTIONS_CSV), header=0)
USER_ITEM_MATRIX_DF = pd.read_pickle(DATA_PATH.joinpath(USER_ITEM_MATRIX_P))



NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Bank Customer Complaints", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)


LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="User Selection", className="display-5"),
        html.Hr(className="my-2"),

        html.Label("Select a bank", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="bank-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
        html.Label("Select number of recommendations.", className="lead"),
        dcc.Slider(
            id="recommend-number",
            min=1,
            max=100,
            step=1,
            marks={
                10: "10",
                20: "",
                30: "30",
                40: "",
                50: "50",
                60: "",
                70: "70",
                80: "",
                90: "90",
                100: "",
            },
            value=10,
        ),
      
    ]
)


WORDCLOUD_PLOT = [
    dbc.CardHeader(html.H5("Most frequently tokenized words in user recommendation")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-frequencies",
                            children=[dcc.Graph(id="frequency_figure")],
                            type="default",
                        )
                    ),
                    dbc.Col(
                        [ dcc.Loading( id="loading-wordcloud",
                                       children=[ dcc.Graph(id="userwordcloud")
                                                ], type="default",
                                     )
                        ]),
                 ])
         ],
         md=8),
]


BODY = dbc.Container(
    [dcc.Tabs([
        dcc.Tab(label='User Recommendation', children=[
          
            dbc.Row(
                [
                    dbc.Col(LEFT_COLUMN, md=4, align="center"),
                    dbc.Col(dbc.Card(TOP_BANKS_PLOT), md=8),
                ],
                style={"marginTop": 30},
            ),
            dbc.Card(WORDCLOUD_PLOTS),
            ]),
            
        dcc.Tab(label='Article Recommendation', children=[])
        ])],
        className="mt-12",
)


reco = Recommender()

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)
app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""



@app.callback(
    [
        Output("userwordcloud", "figure"),
        Output("no-data-alert", "style"),
    ],
    [
        Input("user-dropdown", "value"),
        Input("recommend-number", "value")
    ],
)
def update_wordcloud_plot(value_drop, rec_number):
    """ Callback to rerender wordcloud plot """
    word_text = get_token_words(user, rec_number)
    wordcloud = plotly_wordcloud(word_text)
    alert_style = {"display": "none"}
    if (wordcloud == {}):
        alert_style = {"display": "block"}
    print("redrawing bank-wordcloud...done")
return wordcloud