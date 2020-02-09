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


def flat_arr_df(two_d_data):
    """
    Flatten array/list of arrays/lists and dataframes to lists.

    Input arguments:
        two_d_data: dataframe or array/list of arrays/lists 
                    Example: [[1,2,3],[4,5,6],[7,8,9]]
              
    Output:
        flat_data: List of flattened Input
                   Example: [1,2,3,4,5,6,7,8,9]
    
    """

    if isinstance(two_d_data, (list, np.ndarray)):
        if isinstance(two_d_data[0], (list, np.ndarray)):
            flat_data = [item for sublist in two_d_data for item in sublist]
        else:
            print("Wrong datatype used, cannot flat this object")
            return ""
    elif isinstance(two_d_data, pd.DataFrame):
            flat_data = list(two_d_data.values.flatten())
    
    return flat_data


def return_flatted_f1_prec_recall(Y_pred, Y_test, mac_avg=False):
    """
    Output classification report (f1, precision, recall) for flatted prediction and test data.

    Input arguments:
        Y_pred: dataframe or array/list of arrays/lists 
                    Example: [[1,2,3],[4,5,6],[7,8,9]]
        
        Y_test: dataframe or array/list of arrays/lists 
                    Example: [[1,2,3],[4,5,6],[7,8,9]]
                    
        mac_avg: If True returns F1-Score
              
    Output:
        If mac_avg==False: prints precision recall and f1-score
        If mac_avg==True: returns F1-Score only
    
    """
    flat_Y_pred = flat_arr_df(Y_pred)
    flat_Y_test = flat_arr_df(Y_test)
    if mac_avg:
        return classification_report(flat_Y_pred, flat_Y_test, output_dict=True)["macro avg"]["f1-score"]
    else:
        print(classification_report(flat_Y_pred, flat_Y_test))


#evaluate model and output summarized classification report
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    return_flatted_f1_prec_recall(Y_pred, Y_test)
    pass


# save the resulting model
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