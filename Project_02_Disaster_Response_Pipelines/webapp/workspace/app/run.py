import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# for graph visualisation
from matplotlib import cm
from matplotlib import colors
from scipy import interpolate
from pyvis.network import Network

import os
import sys
sys.path.append('../')
from models.train_classifier import MessageIsQuestion
from sklearn.base import BaseEstimator, TransformerMixin
import nltk


app = Flask(__name__,
            static_folder='static',)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def create_graph(df, path="static/graph_disaster_response.html"):
    '''
    Using PyVis to create Graph Plot with
    16 most frequent Categories
    100 most frequent tokenized words
    frequency weighted Edges
    
    Input Arguments:
        df: DataFrame that has the columns "message","related" and 32 classification categories
        path: Path to save the resulting graph
    Output:
        will save resulting graph in path
    '''
    def prepare_data_graph(df):
        '''
        Preparing DataFrame for Graph Plot:
        - Tokenizing words
        - Explode resulting arrays in seperate rows
        - melting categories into a single column
        
        Input Arguments:
            df: DataFrame that has the columns "message","related" and 35 classification categories
        
        Output:
            df_tok_melt: DataFrame with tokenized messages and melted categories
        '''
        # tokenize messages
        df["tok_message"] = df["message"].apply(tokenize)
        # explode tokenized messages into seperate rows
        df_tok = df.explode("tok_message").drop(columns=["message","original","id"])
        # only take rows that are "related"
        df_tok = df_tok[df_tok["related"] == 1]
        df_tok.drop(columns=["related"], inplace=True)
        # getting an index column that counts from 1 to end
        df_tok = df_tok.reset_index().drop(columns=["index"]).reset_index()
        # extract the columns of the categories
        var_columns = df_tok.columns[2:36]
        # melting 35 categories into a single column
        df_tok_melt = df_tok.melt(id_vars = ['index', 'tok_message'], value_vars=var_columns)
        # take only the rows where the category is 1
        df_tok_melt = df_tok_melt[df_tok_melt["value"] == 1]
        
        return df_tok_melt
    
    df_tok_melt = prepare_data_graph(df)
    
    # prepare interpolation for map values to range between 0 and 1
    cat_interpol = interpolate.interp1d([0, 15], [0,1])

    # init PyVis Network graph with Physic model
    g = Network("1000px", "1500px", notebook=True)
    g.hrepulsion(central_gravity=6.55, spring_length=620, node_distance=465, damping=1)
    
    cat_dict = {}
    node_index = 0
    for cat, value in df_tok_melt["variable"].value_counts().sort_values(ascending=False)[:15].iteritems():
        node_index += 1
        cat_dict[cat] = node_index
        g.add_node(cat_dict[cat], value=value*10000, title="Category {}: {}".format(cat,str(value)), label=cat )
    
    word_dict = {}
    for word, value in df_tok_melt["tok_message"].value_counts().sort_values(ascending=False)[:100].iteritems():
        node_index += 1
        word_dict[word] = node_index
        g.add_node(word_dict[word], value=value*1000, title="Word: {}: {}".format(word,str(value)), color="red",label=word)

    word_per_cat = pd.DataFrame(df_tok_melt.groupby("tok_message")["variable"].value_counts()).T
    edge_dict = {}
    for word in word_dict.keys():
        for cat in word_per_cat[word].columns:
            try:
                g.add_edge(cat_dict[cat], word_dict[word], width= int(word_per_cat[word][cat]["variable"]/50),color = colors.to_hex(cm.get_cmap("rainbow")(cat_interpol(cat_dict[cat]))))
            except:
                pass

    g.save_graph(path)

    
# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    
    graph_path = "static/graph_disaster_response.html"
    if not os.path.exists(graph_path):
            create_graph(df)
            
    graphname = graph_path
    print(graphname)
    # create visuals
    
    cols = df.iloc[:,5:].columns
    df_melt = df.melt(id_vars = ['message'], value_vars=cols)
    category_counts = df_melt[df_melt["value"] == 1]["variable"].value_counts()
    category_names = list(category_counts.index)
    


    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, graphname=graphname)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )





def main():
    app.run(host='0.0.0.0', port=8000, debug=True)


if __name__ == '__main__':
    main()