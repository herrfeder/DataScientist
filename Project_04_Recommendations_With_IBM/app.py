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
import dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import random
from dash.dependencies import Output, Input, State
from dateutil import relativedelta
from plotlywordcloud import plotly_wordcloud
from recommendengine import Recommender


DATA_PATH = pathlib.Path(__file__).parent.resolve()
EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
USER_INTERACTIONS_CSV = "data/user-item-interactions.csv"
USER_ITEM_MATRIX_P = "data/user_item_matrix.p"
UDACITY_LOGO = "https://cdn.worldvectorlogo.com/logos/udacity.svg"


reco = Recommender(df_path=DATA_PATH.joinpath(USER_INTERACTIONS_CSV), 
                   matrix_path=DATA_PATH.joinpath(USER_ITEM_MATRIX_P))

USERS = reco.ra.get_all_users()
ARTICLES = reco.ra.get_all_articles()

user_labels = []
for user in USERS:
    user_labels.append(  {"label": user,
                          "value": user,
})
    
user_labels.append({"label": "New", "value": "new"})


article_labels = []
for article in ARTICLES:
    article_labels.append(  {"label": str(article),
                             "value": article,
})

example_images = ["/static/images/{}".format(x) for x in os.listdir("static/images")]

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=UDACITY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Recommendations with IBM Project", className="ml-2")),
                    dbc.Col(dbc.NavLink("by David Lassig",
                                        href="https://www.linkedin.com/in/davidlassig/",
                                        style={"font-color": "white", "fontSize": 10, "font-weight": "lighter"}, 
                                        className="ml-2")),
                    dbc.Col(dbc.NavLink("Github Repo", href="#")),
                    dbc.Col(dbc.NavLink("Udacity", href="#")),

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


LEFT_USER_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="User Selection", className="display-5"),
        html.Hr(className="my-2"),

        html.Label("Select Type of Recommendation", style={"marginTop": 50}, className="lead"),
        html.P(
            "(New User will only get articles based of popularity)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            options=[{"label":"Collaborative Filtering",
                      "value":"collab"},
                     {"label":"Singular Value Decomposition (SVD)",
                      "value":"svd"}
            ],        
            value="collab",
            id="reco-drop-user", 
            clearable=False, 
            style={"marginBottom": 1, "font-size": 12}
        ),
        
        html.Label("Select a User", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            options=user_labels,
            value="new",
            id="user-drop", 
            clearable=False, 
            style={"marginBottom": 50, "font-size": 12}
        ),
        html.Label("Select number of recommendations.", className="lead"),
        dcc.Slider(
            id="recommend-number-user",
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

LEFT_ARTICLE_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="Article Selection", className="display-5"),
        html.Hr(className="my-2"),

        html.Label("Select Type of Recommendation", style={"marginTop": 50}, className="lead"),
        html.P(
            "(For articles we only have content based recommendation)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            options=[{"label":"Content Based",
                      "value":"content"},
            ],        
            value="content",
            id="reco-drop-art", 
            clearable=False, 
            style={"marginBottom": 1, "font-size": 12}
        ),
        
        html.Label("Select a Article", style={"marginTop": 50}, className="lead"),
        html.P(
            "(Will return articles with highest similarity)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            options=article_labels,
            value=article_labels[0]["value"],
            id="article-drop", 
            clearable=False, 
            style={"marginBottom": 50, "font-size": 12}
        ),
        html.Label("Select number of recommendations.", className="lead"),
        dcc.Slider(
            id="recommend-number-art",
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


def get_art_con(title, popularity=-1, similarity=-1):
    
    if (popularity!=-1) and (similarity!=-1):
        badge_list = [dbc.Badge("Popularity: "+str(popularity), color="light", className="mr-1"),
                      dbc.Badge("Similarity: "+str(similarity), color="light", className="mr-1")]
    elif (similarity!=-1):
        badge_list = [dbc.Badge("Similarity: "+str(similarity), color="light", className="mr-1")]
    elif (popularity!=-1):
        badge_list = [dbc.Badge("Popularity: "+str(popularity), color="light", className="mr-1")]
    else:
        badge_list = [dbc.Badge("Similarity: "+str(similarity), color="light", className="mr-1 invisible")]
    
    card = dbc.Card([
            dbc.CardImg(src=random.choice(example_images), top=True),
            dbc.CardBody(
            [
                html.P(
                    title,
                    className="card-text",
                ),
                dbc.Col(children=[
                    dbc.Row(children=badge_list, style={"margin-bottom": 5}),
                    dbc.Row(dbc.Button("Read Article (dummy)", color="info", size="sm", style={"font-size": "0.8em"}))
                ])
            ])
    ], style={"height":"275px"})
    
    return card


def populate_reco_articles(art_dict_list):
    row_children = []
    body = []
    for index, item in enumerate(art_dict_list):
        row_children.append(dbc.Col(
            get_art_con(item.get("title",""), 
                        item.get("popularity",-1),
                        item.get("similarity", -1))
            ))
        if ((index+1) % 3 == 0) and (index!=0):
            body.append(dbc.Row(children=row_children, 
                                style={"margin-bottom":8}))
            row_children = []
            
    for i in range(len(row_children),3):
        row_children.append(dbc.Col())
        
    body.append(dbc.Row(children=row_children,
                        style={"margin-bottom":8}))
    return body
        
    


RECOMMENDATIONS_USER = [
    dbc.CardHeader(html.H5("10 Best Recommendations for User New (Popularity Based)", id="reco-title-user")),
    dcc.Loading( id="loading-rec-user", children=[
        dbc.CardBody(children=[], id="reco-body-user", style={"overflow-y": "scroll", "height": "580px"} )])
]

RECOMMENDATIONS_ARTICLE = [
    dbc.CardHeader(html.H5("10 Best Recommendations for User New (Popularity Based)", id="reco-title-art")),
    dcc.Loading( id="loading-rec-art", children=[
        dbc.CardBody(children=[], id="reco-body-art", style={"overflow-y": "scroll", "height": "580px"} )])
]

WORDCLOUD_PLOT_USER = [
    dbc.CardHeader(html.H5("Most frequently tokenized words in user recommendation")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert-user",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [dbc.Col(
                        [ dcc.Loading( id="loading-wordcloud-user",
                                       children=[ dcc.Graph(id="userwordcloud-user")
                                                ], type="default",
                                     )
                        ]),
                 ])
         ]),
]

WORDCLOUD_PLOT_ARTICLE = [
    dbc.CardHeader(html.H5("Most frequently tokenized words in user recommendation")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert-art",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [dbc.Col(
                        [ dcc.Loading( id="loading-wordcloud-art",
                                       children=[ dcc.Graph(id="userwordcloud-art")
                                                ], type="default",
                                     )
                        ]),
                 ])
         ]),
]


TABLE_VIEW = dash_table.DataTable(id="tableview")

BODY = dbc.Container(
    [dcc.Tabs([
        dcc.Tab(label='Collaboration Based Recommendation', children=[
          
            dbc.Row(
                [
                    dbc.Col(LEFT_USER_COLUMN, md=4, align="center"),
                    dbc.Col(dbc.Card(RECOMMENDATIONS_USER), md=8),
                ],
                style={"marginTop": 30},
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(WORDCLOUD_PLOT_USER)),
                    dbc.Col(TABLE_VIEW)
                    
            ])]),
            
        dcc.Tab(label='Content Based Recommendation', children=[
            dbc.Row(
                [
                    dbc.Col(LEFT_ARTICLE_COLUMN, md=4, align="center"),
                    dbc.Col(dbc.Card(RECOMMENDATIONS_ARTICLE), md=8),
                ],
                style={"marginTop": 30},
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(WORDCLOUD_PLOT_ARTICLE)),
            ])
        ])
    ]),
], className="mt-12",)
        




server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=["/static/css/bootstrap.min.css"], server=server)
app.layout = html.Div(children=[NAVBAR, BODY])
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

"""
#  Callbacks
"""



@app.callback(
    [
        Output("userwordcloud-user", "figure"),
        Output("reco-body-user", "children"),
        Output("no-data-alert-user", "style"),
        Output("reco-title-user", "children")
    ],
    [
        Input("user-drop", "value"),
        Input("recommend-number-user", "value"),
        Input("reco-drop-user", "value")
    ],
)
def update_wordcloud_reco(user_id, rec_number, rec_type):
    """ Callback to rerender wordcloud plot """
    if user_id=="new":
        reco_title = "{} Best Recommendations for User {} (Article Popularity)".format(rec_number, user_id)
        user_recs, word_text, rec_ids = reco.ra.get_top_articles(rec_number)
    else:
        reco_title = "{} Best Recommendations for User {} (Similarity of Users)".format(rec_number, user_id)
        user_recs, word_text, rec_ids = reco.make_collab_recs(user_id, rec_number)

    wordcloud = plotly_wordcloud(word_text)
    reco_body = populate_reco_articles(user_recs)
    alert_style = {"display": "none"}
    if (wordcloud == {}):
        alert_style = {"display": "block"}
    
    return (wordcloud, reco_body, alert_style, reco_title)




if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="0.0.0.0")