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

APP_PATH = pathlib.Path(__file__).parent.resolve()

do_big = data_prep_helper.ModelData(chart_col=["Price", "High", "Low", "Price_norm"])
do_small = data_prep_helper.ShiftChartData(chart_col="Price")


BASE_COLUMNS = list(do_small.chart_df.columns)

FORE_DAYS = do_big.get_forecast_dates()


column_small_labels = []
for col in BASE_COLUMNS:
    column_small_labels.append({"label": col,
                                "value": col,
                                })
    
fore_days_labels = []
for day in FORE_DAYS:
    fore_days_labels.append({"label": day,
                             "value": day})
    

acc_str_list = ["A Introduction",
                "B View Data", 
                "C Correlation Analysis",
                "D Causality Analysis",
                "E Model Evaluation",
                "F Forecast"]

acc_slider_list = [["Introduction", "Resources"],
                   ["View Data","Conclusions"],
                   ["Simple Correlation", "Correlation Timeshift", "Conclusions"],
                   ["Seasonal Analysis", "Granger Causality", "Conclusions"],
                   ["ARIMAX", "VAR", "GRU"],
                   ["Forecast", "Chances and next Steps"]]
    
# Load data


GRANGER_PATH = APP_PATH.joinpath("data/granger_causality.csv")

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




LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="User Selection", className="display-5"),
        html.Hr(className="my-2"),
        html.Div(make_items(acc_str_list, acc_slider_list), className="accordion"),   
    ]
)

RIGHT_COLUMN = html.Div(id="right_column", children=[dcc.Loading(id="right_column_loading")])


### A INTRODUCTION ###

INTRODUCTION = html.Div(dcc.Markdown(conclusion_texts.introduction), id="introduction")

RESOURCES = ""


### B VIEW DATA ###

EXP_CHART_PLOT = [dbc.CardHeader(html.H5("Historic Input Datasets")),
              dbc.CardBody(dcc.Loading(dcc.Graph(id="exp_chart_plot",
                           figure=exploratory_fig)))
             ]

VIEW_CONCLUSIONS = ""


### C CORRELATION ANALYSIS ###

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
                                                        figure=ph.return_shift_corr(do_small,
                                                                                    "bitcoin_Price", 
                                                                                    -30, 
                                                                                    dash=True))
                                          )             
                            ])
                        )
             ]

CORR_CONCLUSIONS = html.Div(dcc.Markdown(conclusion_texts.correlation_conclusion), id="corr-conclusions")


### D CAUSALITY ANALYSIS ###

CAUS_SEASONAL_DROPDOWN = html.Div([
                        dcc.Dropdown(id='caus_seasonal_dropdown',
                                     options=column_small_labels,
                                     value="bitcoin_Price")
                     ,], style={"width":"20%"})



CAUS_SEASONAL_PLOT = [dbc.CardHeader(html.H5("Seasonal Decomposition")),
              dbc.CardBody(html.Div(children=[
                                        dbc.Row(children=[
                                                html.Label("Seasonal Decomposition for:",
                                                    style={"padding-left":20,
                                                           "padding": 10}), 
                                                CAUS_SEASONAL_DROPDOWN],
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                            
                                            dcc.Loading(
                                                dcc.Graph(id="caus_seasonal_plot",
                                                          figure=ph.return_season_plot(do_small.chart_df, 
                                                                                       "bitcoin_Price", 
                                                                                       title="", 
                                                                                       dash=True)))
                                         ]))
                     ]

CAUS_GRANGER_PLOT = [dbc.CardHeader(html.H5("Granger Causality with Time Lag of 30 Days")),
                     dbc.CardBody(
                         html.Div(children=[
                                         dbc.Row( 
                                             html.Div(
                                                 dcc.Markdown(conclusion_texts.granger_prob_expl,
                                                              style={"padding-left":20, "padding": 10}), 
                                                 id="granger_expl"),
                                         
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                        dcc.Loading(
                                            dcc.Graph(id="caus_granger_plot",
                                                      figure=ph.return_granger_plot(GRANGER_PATH,  
                                                                                    title="",
                                                                                    colormap="viridis_r",
                                                                                    dash=True)))
                         ]
                                         ))]


### E MODEL EVALUATION ###

### F FORECAST ###

FORE_DAYS_DROPDOWN = html.Div([
                        dcc.Dropdown(id='fore_days_dropdown',
                                     options=fore_days_labels,
                                     value=FORE_DAYS[0])
                     ,], style={"width":"20%"})


FORE_SENTIMENTS = [dbc.CardHeader(html.P("SENTIMENTS")),
                   dbc.CardBody(html.P("test"))]

FORE_TRENDS = [dbc.CardHeader(html.P("TRENDS")),
                   dbc.CardBody(html.P("test"))]

FORE_STOCKS = [dbc.CardHeader(html.P("STOCKS")),
                   dbc.CardBody(html.P("test"))]

FORE_ALL = [dbc.CardHeader(html.H5("Forecasting and Parameters for Day")),
                        dbc.CardBody( 
                            html.Div(children=[
                                        dbc.Row(children=[
                                            html.Label("Column to Fix:",
                                                style={"padding-left":20,
                                                       "padding": 10}), 
                                            FORE_DAYS_DROPDOWN,
                                            html.Label("Past Timeshift:",
                                                style={"padding-left":20,
                                                       "padding": 10}),
                                            ],
                                            align="center",
                                            style={"background-color": "#073642", "border-radius": "0.3rem"}),
                                
                                          dcc.Loading(
                                              dcc.Graph(id="fore_plot")
                                                       
                                          ),
                                          dbc.Row(children=[dbc.Col(FORE_SENTIMENTS),
                                                            dbc.Col(FORE_TRENDS),
                                                            dbc.Col(FORE_STOCKS)])
                            ])
                        )
             ]


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
def show_plot(acc_01, acc_02, acc_03, acc_04, acc_05, acc_06,
              sli_01, sli_02, sli_03, sli_04, sli_05, sli_06,
              figure):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        element_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in element_id):
        if sli_01 == 1:
            return INTRODUCTION
        if sli_01 == 0:
            return "" # needs to be filled with RESOURCES
    elif (acc_str_list[1] in element_id):
        if sli_02 == 1:
            return EXP_CHART_PLOT
        if sli_02 == 0:
            return "" # needs to be filled with VIEW_CONCLUSIONS
    elif (acc_str_list[2] in element_id):
        if sli_03 == 2:
            return CORR_01_CHART_PLOT
        elif sli_03 == 1:
            return CORR_02_CHART_PLOT
        elif sli_03 == 0:
            return CORR_CONCLUSIONS
    elif (acc_str_list[3] in element_id):
        if sli_04 == 2:
            return CAUS_SEASONAL_PLOT
        if sli_04 == 1:
            return CAUS_GRANGER_PLOT
        if sli_04 == 0:
            return ""
        
    elif (acc_str_list[5] in element_id):
        return FORE_ALL


@app.callback(
    Output("fore_plot", "figure"),
    [Input("fore_days_dropdown", "value")], 
    [State("fore_plot", "figure")])
def plot_forecast(curr_day, figure, plot_real_comp=False, conf_interval=False, boll_fore=False, boll_real=False):
    curr_fore, curr_real = do_big.ari_forecast(curr_day)
    return ""

@app.callback(
    Output("caus_seasonal_plot", "figure"),
    [Input("caus_seasonal_dropdown", "value")],
    [State("caus_seasonal_plot", "figure")]
)
def ret_caus_seasonal_plot(dropdown, figure):
    
    return ph.return_season_plot(do_small.chart_df, dropdown, title="", dash=True)
    
    
@app.callback(
    Output("corr_shift_matrix_plot", "figure"),
    [Input("corr_shift_button", "n_clicks")],
    [State("corr_shift_dropdown", "value"), State("corr_shift_slider", "value")]
)
def ret_corr_shift_plot(n, dropdown, slider):
    
    return ph.return_shift_corr(do_small, dropdown, slider, output="single",dash=True)
    
    
@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in acc_str_list],
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list],
    [State(f"collapse-{i}", "is_open") for i in acc_str_list],
)
def toggle_accordion(n1, n2, n3, n4, n5, n6,
                     is_open1, is_open2, is_open3, is_open4, is_open5, is_open6):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if (acc_str_list[0] in button_id) and n1:
        return not is_open1, False, False, False, False, False
    elif (acc_str_list[1] in button_id) and n2:
        return False, not is_open2, False, False, False, False
    elif (acc_str_list[2] in button_id) and n3:
        return False, False, not is_open3, False, False, False
    elif (acc_str_list[3] in button_id) and n4:
        return False, False, False, not is_open4, False, False
    elif (acc_str_list[4] in button_id) and n5:
        return False, False, False, False, not is_open5, False
    elif (acc_str_list[5] in button_id) and n6:
        return False, False, False, False, False, not is_open6
    
    return False, False, False, False, False, False        



output=[]
output.extend([Output(f"slidersub-{i}", "children") for i in acc_str_list])

state=[]
state.extend([State(f"slidersub-{i}", "children") for i in acc_str_list])

@app.callback(
    output,
    acc_input,
    state,
)    
def update_sub(acc_01, acc_02, acc_03, acc_04, acc_05, acc_06,
              sli_01, sli_02, sli_03, sli_04, sli_05, sli_06,
              slisub_01, slisub_02, slisub_03, slisub_04, slisub_05, slisub_06):

    ctx = dash.callback_context

    if not ctx.triggered:
        return "", "", "", "", "", ""
    else:
        element_id = ctx.triggered[0]["prop_id"].split(".")[0]

        
    if (acc_str_list[0] in element_id):
        return "", "", "", "", "", ""
    elif (acc_str_list[1] in element_id):
        if sli_02 == 1:
            return "Show all Input Time Series", "", "", "", "", ""
        elif sli_02 == 0:
            return "Show all Input Time Series","","", "", "", ""
    elif (acc_str_list[2] in element_id):
        if sli_03 == 2:
            return "","Simple Correlation between all Input Time Series", "", "", "", ""
        elif sli_03 == 1:
            return "","Correlation between timeshifted Time Series", "", "", "", ""
        elif sli_03 == 0:
            return "","Conclusions resulting from Correlation Analysis", "", "", "", ""
    elif (acc_str_list[3] in element_id):
        if sli_04 == 1:
            return "","","blah", "", "", ""
        elif sli_04 == 0:
            return "","","blubb", "", "", ""
    elif (acc_str_list[4] in element_id):
        return "", "", "", "", "", ""
    elif (acc_str_list[5] in element_id):
        return "", "", "", "", "", ""

    
    
@app.callback(
    [Output(f"spandot-{i}", "style") for i in acc_str_list],
    [Input(f"group-{i}-toggle", "n_clicks") for i in acc_str_list],
    [State(f"spandot-{i}", "style") for i in acc_str_list],
)
def toggle_active_dot(n1, n2, n3, n4, n5, n6, 
                      active1, active2, active3, active4, active5, active6):
    
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
        return sty_a, sty_na, sty_na, sty_na, sty_na, sty_na
    elif (acc_str_list[1] in button_id) and n2:
        return sty_na, sty_a, sty_na, sty_na, sty_na, sty_na
    elif (acc_str_list[2] in button_id) and n3:
        return sty_na, sty_na, sty_a, sty_na, sty_na, sty_na
    elif (acc_str_list[3] in button_id) and n4:
        return sty_na, sty_na, sty_na, sty_a, sty_na, sty_na 
    elif (acc_str_list[4] in button_id) and n5:
        return sty_na, sty_na, sty_na, sty_na, sty_a, sty_na 
    elif (acc_str_list[5] in button_id) and n6:
        return sty_na, sty_na, sty_na, sty_na, sty_na, sty_a
    
    return sty_na, sty_na, sty_na, sty_na, sty_na, sty_na 
           

if __name__ == "__main__":
    app.run_server(debug=True,  port=8050, host="0.0.0.0")
