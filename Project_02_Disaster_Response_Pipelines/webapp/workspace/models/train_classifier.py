# sys internals
import sys

# basic data libraries
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

# scikit-learn modules for pipelining, transformation, model fitting and classification
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# scikit-learn model evaluation
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

# nltk-modules for text processing, tokenizing and lemmatizing
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download relevant ntlk packages
nltk.download(["punkt", "stopwords", "wordnet"])

# pickle for python object serialization and storing
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.loc[:,"message"]
    Y = df.iloc[:,4:40]
    
    return (X,Y,list(Y.columns))


def tokenize(text):
    """
    Tokenize, lemmatize, lower and remove punctuation of input text.

    Input arguments:
        text: Single string with input text 
              Example: 'For today:= this is, a advanced _ example #- String!'
              
    Output:
        output: List of processed string
                Example: ['today', 'advanced', 'example', 'string']
        
    """
    # set text to lower case and remove punctuation
    text = re.sub("[\W_]", " ", text)
    text= text.lower()

    # tokenize words 
    tokens = word_tokenize(text)
    
    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # init and remove stopwords
    stop_words = set(stopwords.words('english'))
    output = [lemmatizer.lemmatize(w) for w in tokens if not w in stop_words]

    return output


def build_model(parameters={}):
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100,
                                                             min_samples_split=2)))
    ])
    
    return pipeline
    


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()