import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
                             y=df['gold_Price_norm'],
                             name="GOLD Normed Close"), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['amazon_Price_norm'],
                             name="AMAZON Normed Close"), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, 
                             y=df['alibaba_Price_norm'],
                             name="ALIBABA Normed Close"), row=2, col=1)

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


def plot_val_heatmap(df, title="", dash=False):
    
    coordinates = df.values.tolist()
    columns = list(df.columns)
    index = list(df.index)

    trace1 = {
            "type": "heatmap", 
            "x": columns, 
            "y": index, 
            "z": coordinates,
            "colorscale": 'viridis'
            }

    data = trace1
    layout = {"title": title,
              "height": 1000}
    fig = go.Figure(dict(data=data, layout=layout))
    
    if dash:
        return apply_layout(fig, title, height=800)
    else:
        fig.show()
        
        
def return_shift_corr(do, fixed="bitcoin_Price", shift=-30, dash=False):
    
    do.fixed_cols = fixed
    corr = do.single_shift(shift).corr()
    
    return plot_val_heatmap(corr, dash=True)
    
