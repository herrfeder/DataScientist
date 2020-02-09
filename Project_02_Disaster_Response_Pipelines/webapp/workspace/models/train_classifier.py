# sys internals
import sys

# basic data libraries
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

# scikit-learn modules for pipelining, transformation, model fitting and classification
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


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


class MessageIsQuestion(BaseEstimator, TransformerMixin):
    '''
    Extract messages that starts with question word or ends with question mark and
    return DataFrame with 1's for True and 0's for False
    '''
    def __init__(self):
        '''
        Create the Regex Variable for checking
        '''
        # typical englisch question words
        question_words = ["what", "when", "do", "is", "who", "which", "where", "why", "how"]
        # matches question words at beginning of text or questionmarks at the end
        question_reg = "(^"+"("+"|".join(question_words)+")|(\?)$)"
        self.q_reg = re.compile(question_reg)
    
    def message_question(self, text):
        '''
        Will get on text message per execution. After Tokenizing by sentences, it will return 1
        on matching the regex question_reg or 0 for not matching.
        
        Input Arguments:
            text: Single String with message
            
        Output:
            output: Returns 1 if text includes Question and returns 0 when Question doesn't include question
        
        '''
        # tokenize by sentences
        sentence_list =  nltk.sent_tokenize(text)
        for sentence in sentence_list:
            # find pattern question_reg in each sentence
            sentence = sentence.lower()
            if self.q_reg.match(sentence):
                return 1
            else:
                return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        '''
        Will go through single column DataFrame and applies message_question to every message in the row.
        The returning DataFrame holds 1's or 0's whether the message includes question or not.
        
        Input Arguments:
            X: DataFrame with single column or array/list
        
        Output:
            output: Cleaned DataFrame with 1's and 0's for messages
        '''
        
        # apply message_question function to all values in X
        X_tagged = pd.Series(X).apply(self.message_question)
        
        # clean the resulting Dataframe that it can be processed through the pipeline
        df_t = pd.DataFrame(X_tagged)
        df_t.fillna(0, inplace=True)
        df_t = df_t.astype(int)
        
        return df_t


def build_model():
    '''
    Pipeline with FeatureUnion that feeds the output of TfidfTransformer as features and the message question
    feature into the model.
    
    Input Arguments:
    
    Output Arguments:
        output: Returns Pipeline
    '''
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('message_question', MessageIsQuestion())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=300,
                                                             min_samples_split=3,
                                                             criterion="gini"))
    )])
    
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