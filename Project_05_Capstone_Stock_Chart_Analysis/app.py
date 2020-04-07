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

import plot_helper as ph
import data_prep_helper
import conclusion_texts

from dash.dependencies import Input, Output, State
from plotly import tools

do_big = data_prep_helper.ValidateChartData(chart_col=["Price", "High", "Low", "Price_norm"])
do_small = data_prep_helper.ValidateChartData(chart_col="Price")


BASE_COLUMNS = list(do_small.chart_df.columns)


column_small_labels = []
for col in BASE_COLUMNS:
    column_small_labels.append({"label": col,
                                "value": col,
                                })

acc_str_list = ["View Data", 
                "Correlation Analysis",
                "Causality Analysis"]

acc_slider_list = [["blah","blubb"],
                   ["Simple Correlation", "Correlation Timeshift", "Conclusions"],
                   ["Granger Causality", "Conclusions"]]
    
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


def make_items(acc_str_list, acc_slider_list):
    card_list = []
    for acc_str, acc_slider in zip(acc_str_list, acc_slider_list):
        card_list.append(dbc.Card(
            [
                dbc.CardHeader(
                        dbc.Row([
                            html.Span(id=f"spandot-{acc_str}",
                                      style={"height": "15px", 
                                             "width": "15px", 
                                             "background-color": "#bbb", 
                                             "border-radius": "50%",
                                             "padding-left": 20
                                             }
                                     ),
                        dbc.Button(
                            f"{acc_str}",
                            color="link",
                            id=f"group-{acc_str}-toggle",
                            style={"padding-top":10}

                            )
                        ],style={"display":"inline-flex", 
                                 "align-items":"center",
                                 "padding-left":20} 
                        ) 
                    ),


                dbc.Collapse(
                    html.Div(children=[dbc.Col(
                                dcc.Slider(
                                        id = f"slider-{acc_str}",
                                        updatemode = "drag",
                                        vertical = True,
                                        step=None,
                                        marks = {index: {"label":"{}".format(name),
                                                         "style": {"color": "#2AA198"}
                                                        } for index,name in enumerate(acc_slider[::-1])},
                                        min=0,
                                        max=len(acc_slider)-1,
                                        value=len(acc_slider)-1,
                                        verticalHeight=len(acc_slider)*50)
                            ),
                            dbc.Col(html.P(id=f"slidersub-{acc_str}", 
                                           style={"color":"orange"}), 
                                    id=f"slider-content-{acc_str}")],
                            style={"padding":10, "padding-left":20}), id=f"collapse-{acc_str}"       
                )
            ])
    )
         
    return card_list
            
exploratory_fig = ph.exploratory_plot(do_big.apply_boll_bands("bitcoin_hist",
                                                              append_chart=False), title="", dash=True)
corr_01_matrix_plot = ph.plot_val_heatmap(do_small.chart_df.corr(), 
                                          title="", 
                                          dash=True)

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


CORR_SHIFT_DROPDOWN = html.Div([
                        dcc.Dropdown(id='corr_shift_dropdown',
                                     options=column_small_labels,
                                     value="bitcoin_Price")
                     ,], style={"width":"20%"})

CORR_SHIFT_SLIDER = html.Div(children=[
                        dcc.Slider(
                            id='corr_shift_slider',
                            updatemode = "drag",
                            marks={day_shift: {
                                            "label": str(day_shift),
                                            "style": {"color": "#7fafdf"},
                                        }
                                        for day_shift in range(-50,5,5) },
                            min=-50,
                            max=0,
                            step=1,
                            value=-30,),
                    ],style={"width":"40%"})

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="User Selection", className="display-5"),
        html.Hr(className="my-2"),
        html.Div(make_items(acc_str_list, acc_slider_list), className="accordion"),   
    ]
)

RIGHT_COLUMN = html.Div(id="right_column", children=[dcc.Loading(id="right_column_loading")])

EXP_CHART_PLOT = [dbc.CardHeader(html.H5("Historic Input Datasets")),
              dbc.CardBody(dcc.Loading(dcc.Graph(id="exp_chart_plot",
                           figure=exploratory_fig)))
             ]

CORR_01_CHART_PLOT = [dbc.CardHeader(html.H5("Correlation Matrix for all Input Time Series")),
              dbc.CardBody(dcc.Loading(dcc.Graph(id="corr_01_matrix_plot",
                           figure=corr_01_matrix_plot)))
             ]

CORR_02_CHART_PLOT = [dbc.CardHeader(html.H5("Correlation Matrix with User-Defined Time Shift")),
                        dbc.CardBody( 
                            html.Div(children=[
                                        dbc.Row(children=[
                                            html.Label("Column to Fix:",
                                                style={"padding-left":20,
                                                       "padding": 10}), 
                                            CORR_SHIFT_DROPDOWN,
                                            html.Label("Past Timeshift:",
                                                style={"padding-left":20,
                                                       "padding": 10}),
                                            CORR_SHIFT_SLIDER,
                                            dbc.Button("Run Correlation", 
                                                       id="corr_shift_button", 
                                                       className="btn btn-success")],
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                
                                          dcc.Loading(
                                              dcc.Graph(id="corr_shift_matrix_plot",
                                                        figure=ph.return_shift_corr(do_small,"bitcoin_Price", -30, dash=True))
                                          )             
                            ])
                        )
             ]

CORR_CONCLUSIONS = html.Div(dcc.Markdown(conclusion_texts.correlation_conclusion), id="corr-conclusions")

TABLE_VIEW = [dbc.CardHeader(html.H5("10 Most Similiar Users")),
              dbc.CardBody(dcc.Loading(dash_table.DataTable(id="usertableview")))
                ]

BODY = dbc.Container([
            dbc.Row(
                [
                    dbc.Col(LEFT_COLUMN, md=3),
                    dbc.Col(RIGHT_COLUMN, md=9),
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
app.config['suppress_callback_exceptions'] = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

server = app.server

acc_input = [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list]
acc_input.extend([Input(f"slider-{i}", "value") for i in acc_str_list])


@app.callback(
    Output("right_column_loading", "children"),
    acc_input,
    [State("right_column_loading", "children")],
)    
def show_plot(acc_01, acc_02, acc_03, 
              sli_01, sli_02, sli_03, 
              figure):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "",""
    else:
        element_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in element_id):
        return EXP_CHART_PLOT
    elif (acc_str_list[1] in element_id):
        if sli_02 == 2:
            return CORR_01_CHART_PLOT
        elif sli_02 == 1:
            return CORR_02_CHART_PLOT
        elif sli_02 == 0:
            return CORR_CONCLUSIONS

        
@app.callback(
    Output("corr_shift_matrix_plot", "figure"),
    [Input("corr_shift_button", "n_clicks")],
    [State("corr_shift_dropdown", "value"), State("corr_shift_slider", "value")]
)
def ret_corr_shift_plot(n, dropdown, slider):
    
    return ph.return_shift_corr(do_small, dropdown, slider, dash=True)
    
    
@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in acc_str_list],
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list],
    [State(f"collapse-{i}", "is_open") for i in acc_str_list],
)
def toggle_accordion(n1, n2, n3, is_open1, is_open2, is_open3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in button_id) and n1:
        return not is_open1, False, False
    elif (acc_str_list[1] in button_id) and n2:
        return False, not is_open2, False
    elif (acc_str_list[2] in button_id) and n3:
        return False, False, not is_open3
    return False, False, False        



output=[]
output.extend([Output(f"slidersub-{i}", "children") for i in acc_str_list])

state=[]
state.extend([State(f"slidersub-{i}", "children") for i in acc_str_list])

@app.callback(
    output,
    acc_input,
    state,
)    
def update_sub(acc_01, acc_02, acc_03, 
              sli_01, sli_02, sli_03, 
              slisub_01, slisub_02, slisub_03):

    ctx = dash.callback_context

    if not ctx.triggered:
        return "",""
    else:
        element_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in element_id):
        return "Show all Input Time Series","",""
    elif (acc_str_list[1] in element_id):
        if sli_02 == 2:
            return "","Simple Correlation between all Input Time Series",""
        elif sli_02 == 1:
            return "","Correlation between timeshifted Time Series",""
        elif sli_02 == 0:
            return "","Conclusions resulting from Correlation Analysis",""

    
    
@app.callback(
    [Output(f"spandot-{i}", "style") for i in acc_str_list],
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list],
    [State(f"spandot-{i}", "style") for i in acc_str_list],
)
def toggle_active_dot(n1, n2, n3, active1, active2, active3):
    
    sty_na={"height": "15px", 
           "width": "15px", 
           "background-color": "#bbb", 
           "border-radius": "50%",
            }
    
    sty_a={"height": "15px", 
           "width": "15px", 
           "background-color": "#00FF00", 
           "border-radius": "50%",
            }
    
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in button_id) and n1:
        return sty_a, sty_na, sty_na
    elif (acc_str_list[1] in button_id) and n2:
        return sty_na, sty_a, sty_na
    elif (acc_str_list[2] in button_id) and n3:
        return sty_na, sty_na, sty_a
    return sty_na, sty_na, sty_na 
           

if __name__ == "__main__":
    app.run_server(debug=True,  port=8050, host="0.0.0.0")
