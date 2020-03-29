import plotly.graph_objects as go

def plot_val_heatmap(df, title):
    
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
              "width": 1200,
              "height": 1000}
    fig = go.Figure(dict(data=data, layout=layout))
    
    return fig.show()