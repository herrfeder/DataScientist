# import libraries

import sys
import os
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Reading the input files into dataframes and merge into a single df.
    
    Input Variables:
        message_filepath: String that holds relative filepath for messages
        categories_filepath: String that holds relative filepath for categories
        
    Output Variables:
        df: Inner Merged DataFrame that holds messages and categories
    
    '''
    
    # reading messages and categories into dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on="id", how="inner")

    return df


def clean_data(df):
    '''
    Wrangling and cleaning the df that it holds only our features and target variables.
    
    Input Variables:
        df: Inner Merged DataFrame with messages and categories
        
    Output Variables:
        df: Cleaned df that holds
                - column "message" with unprocessed string messages as Feature
                - 36 columns with target categories in the value space [0,1]
                - all duplicates are removed
                - Category Values other than {0,1} will be removed
    
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: re.sub("-[0-1]", "",x))
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: re.sub(".*-","",x))
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    
    # drop the original categories column from `df`
    df.drop(columns="categories", inplace=True, errors="ignore")
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # remove values other than {0,1}
    df = df[~(df["related"] == 2)]
    
    # add additional column that holds the melted category
    
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()