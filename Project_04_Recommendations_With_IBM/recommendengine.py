import pathlib
import numpy as np
import pandas as pd
from operator import itemgetter
from scipy.sparse.linalg import svds

import sys # can use sys to take command line arguments
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import nltk

from IPython.core import debugger
debug = debugger.Pdb().set_trace

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("wordnet")  
    

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, df_path, matrix_path):
        '''
        what do we need to start out our recommender system
        '''
        self.rp = RecommenderPreperation(df_path, matrix_path)
        self.df, self.user_item = self.rp.get_datasets()
        
        self.ra = RecommenderAnalysis(self.df, self.user_item)


    def fit_svd(self, k=20):
        '''
        fit the recommender to your dataset and also have this save the results
        to pull from when you need to make predictions
        '''
        # restructure with k latent features
        u, s, vt = svds(self.user_item, k)
        
        # take dot product
        user_item_est = np.around(np.dot(np.dot(u, np.diag(s)), vt))
        
        # convert prediction array to dataframe
        self.preds_df = pd.DataFrame(user_item_est, columns = self.user_item.columns)
        
        
        
    def make_svd_recs(self,):
        '''
        given a user id or a movie that an individual likes
        make recommendations
        '''
        pass
    
    
    
        
        
    def make_collab_recs(self, user_id, m=10):
        """
        Loops through the users based on closeness to the input user_id
        For each user - finds articles the user hasn't seen before and provides them as recs
        Does this until m recommendations are found
        
        INPUT:
        user_id - (int) a user id
        m - (int) the number of recommendations you want for the user

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title

        """
        # create Series with number of interactions per user
        article_interacts = pd.DataFrame(self.df["article_id"].value_counts())

        # get the articles that were already read by user
        own_art_ids, article_names = self.ra.get_user_articles(user_id)

        # get most similiar users
        neighbors_df = self.ra.get_top_sorted_users(user_id)

        # go through neighbors until we got the wanted amount m of recommendations
        recs=[]
        for index, vals in neighbors_df.iterrows():

            # get the read articles by current neighbor 
            article_ids, article_names = self.ra.get_user_articles(vals["neighbor_id"])

            # get number of interactions per article and put into tuple: [(num_interactions, article_id),...]
            article_in_ids  =  [(article_interacts.loc[float(x)].item(), x) for x in article_ids]

            # sort list of tuples by number of interactions and extract list with only id's
            article_in_ids.sort(key=itemgetter(0), reverse=True)
            sorted_article_ids = [tupl[1] for tupl in article_in_ids]

            # go through sorted article_id's and if not already collected and not in articles of user append to recs
            for a_id in sorted_article_ids:
                if (not a_id in own_art_ids) and (not a_id in recs):
                    recs.append(a_id)
                    if len(recs) == m:
                        return recs, self.ra.get_article_names(recs)
                    
                    
            user_recs = []
            for rec, rec_name in zip(recs, self.ra.get_article_names(recs)):
                user_recs.append((
                    rec,
                    rec_name))

        return user_recs, self.ra.get_token_texts(recs)
    
    
    def make_content_recs(self, article_id, m=10, df_ext=""):
        """
        Build Intersections of tokenized string lists over all entries of dataset with given article_id.
        Sort them by frequency and return sorted list with most similiar articles.

        INPUT:
            article_id: article_id for article to make content based recommendations for

        OUTPUT:
            content_articles: tuple with recommended articles, intersections and number of total intersectioned words
        """

        # return df with column with tokenized strings
        if df_ext:
            df_tok = df_ext.copy()
        else:
            df_tok = self.df.copy()
        
        df_tok = df_tok.drop_duplicates(subset="title")
        # accept different types of article_id
        if isinstance(article_id,str):
            article_id = int(article_id.split(".")[0])

        elif isinstance(article_id, float):
            article_id = int(article_id)

        # store the tokenized strings for the article with the given article_id
        own_title_toks = df_tok[df_tok["article_id"] == article_id]["title_tokens"].item()

        # build list of tuple with the (article_id,
        #                               the name of the article,
        #                               the intersection of title tokens,
        #                               sum of intersections)
        content_articles = []
        article_ids = []
        for index, row in df_tok.iterrows():
            set_toks = set(own_title_toks).intersection(set(row["title_tokens"]))

            content_articles.append((
                row["article_id"],
                row["title"],
                set_toks,
                len(set_toks)
            ))

            article_ids.append(row["article_id"])
            
        # sort list of tuples by sum of intersections
        content_articles.sort(key=itemgetter(3), reverse=True)

        return content_articles[1:m+1], self.ra.get_token_texts(article_ids[1:m+1])    


class RecommenderAnalysis():
    
    def __init__(self, df, user_item_matrix):
        
        self.df = df
        self.user_item = user_item_matrix
    
    def get_all_users(self):
        return self.df["user_id"].unique().tolist()
    
    
    
    def get_top_articles(self, n):
        '''
        INPUT:
        n - (int) the number of top articles to return
        df - (pandas dataframe) df as defined at the top of the notebook 

        OUTPUT:
        top_articles - (list) A list of the top 'n' article titles 

        '''
        top_articles_idx = self.df["article_id"].value_counts().nlargest(n).index.tolist()
        
        user_recs = []
        for idx, idx_name in zip(top_articles_idx, self.get_article_names(top_articles_idx)):
                user_recs.append((
                    idx,
                    idx_name))

        return (user_recs, self.get_token_texts(top_articles_idx), idx) # Return the top article ids
    
        
    def get_article_names(self, article_ids):
        """
        INPUT:
        article_ids - (list) a list of article ids
        df - (pandas dataframe) df as defined at the top of the notebook

        OUTPUT:
        article_names - (list) a list of article names associated with the list of article ids 
                        (this is identified by the title column)
        """
        if isinstance(article_ids[0],str):
            article_ids = [int(x.split(".")[0]) for x in article_ids]

        article_names = self.df[self.df["article_id"].isin(article_ids)]["title"].unique().tolist()

        return article_names  # Return the article names associated with list of article ids

    
    def get_token_texts(self, article_ids):
        """
        INPUT:
        article_ids - (list) a list of article ids
        df - (pandas dataframe) df as defined at the top of the notebook

        OUTPUT:
        article_names - (list) a list of article names associated with the list of article ids 
                        (this is identified by the title column)
        """
        if isinstance(article_ids[0],str):
            article_ids = [int(x.split(".")[0]) for x in article_ids]

        df = self.df.drop_duplicates(subset="title")
        
        title_tokens = df[df["article_id"].isin(article_ids)]["title_tokens"].tolist()
        
        token_text = " ".join([item for sublist in title_tokens for item in sublist])

        return token_text  # Return the article names associated with list of article ids


    def get_user_articles(self, user_id):
        """
        Provides a list of the article_ids and article titles that have been seen by a user

        INPUT:
        user_id - (int) a user id
        user_item - (pandas dataframe) matrix of users by articles: 
                    1's when a user has interacted with an article, 0 otherwise

        OUTPUT:
        article_ids - (list) a list of the article ids seen by the user
        article_names - (list) a list of article names associated with the list of article ids 
                        (this is identified by the doc_full_name column in df_content)
        """

        article_ids = self.user_item.loc[user_id][self.user_item.loc[user_id] > 0].index.tolist()

        article_ids = [str(x) for x in article_ids]

        return article_ids, self.get_article_names(article_ids)  # return the ids and names
        
    
    def get_user_interacts(self, user_id):
        """
        """

        user_interacts = self.df[self.df.loc["user_id"] == user_id]["user_id"].count()

        return user_interacts
    
    
    def get_top_sorted_users(self, user_id):
        """
        Sort the neighbors_df by the similarity and then by number of interactions where 
        highest of each is higher in the dataframe.
        
        INPUT:
        user_id - (int)
        df - (pandas dataframe) df as defined at the top of the notebook 
        user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise


        OUTPUT:
        neighbors_df - (pandas dataframe) a dataframe with:
                        neighbor_id - is a neighbor user_id
                        similarity - measure of the similarity of each user to the provided user_id
                        num_interactions - the number of articles viewed by the user
        """

        # create Series with number of interactions per user
        user_interacts = self.df["user_id"].value_counts()

        # recycle find_similiar users and get Series with index and values (similarity) for specific user
        neighbors_df = pd.DataFrame(self.find_similar_users(user_id, mode="index_value"))

        # parse number of interactions into the new neighbors_df
        neighbors_df["num_interactions"] = [user_interacts.loc[x] for x in neighbors_df.index]

        # create new continous index and rename columns to useful names
        neighbors_df = neighbors_df.reset_index().rename(columns={"user_id":"neighbor_id",
                                                                  user_id:"similarity"})

        # sorting neighbors_df by similarity and num_interactions
        neighbors_df = neighbors_df.sort_values(by=["similarity", "num_interactions"], ascending=False)

        return neighbors_df  # Return the dataframe specified in the doc_string

    
    def find_similar_users(self, user_id, sim_level=20, mode="index"):
        '''
        Computes the similarity of every pair of users based on the dot product
        Returns an ordered list.

        INPUT:
        user_id - (int) a user_id
        user_item - (pandas dataframe) matrix of users by articles: 
                    1's when a user has interacted with an article, 0 otherwise
        sim_level - at least level of similarity in percent 

        OUTPUT:
        similar_users - (list) an ordered list where the closest users (largest dot product users)
                        are listed first

        '''
         # the user_id is our user_idx we will be using
        user_idx = user_id

        # creating the dot product to get a symmetric matrix with user_id's in row and columns and the values are the similarities
        user_user_dot = self.user_item.dot(np.transpose(self.user_item))

        # find the most similiar users by at least a similarity level of sim_level
        idx_and_value = user_user_dot[user_user_dot[user_idx] >= ((np.max(user_user_dot[user_idx])/100)*sim_level)][user_idx]

        # sort by similarity and convert to simple list
        most_similiar_user_ids = idx_and_value.sort_values(ascending=False)
        if mode == "index":
            most_similiar_user_ids = most_similiar_user_ids.index.tolist()
            # remove own user_id
            most_similiar_user_ids.remove(user_idx)

            return most_similiar_user_ids
        elif mode=="index_value":
            return most_similiar_user_ids
    
    
class RecommenderPreperation():
    
    def __init__(self, df_path, matrix_path):
        
      
        self.df = pd.read_csv(df_path, header=0)
        # load user_item_matrix from pickle to save resources
        self.user_item_matrix = pd.read_pickle(matrix_path)
        
        # prepare dataframe user_id
        del self.df["Unnamed: 0"]
        email_encoded = self.email_mapper()
        del self.df["email"]
        self.df["user_id"] = email_encoded
        
        # create article tokens
        self.df["title_tokens"] = self.get_tokenized_articles_df()

    def get_datasets(self):
        
        return self.df, self.user_item_matrix
    
    
    def email_mapper(self):
        coded_dict = dict()
        cter = 1
        email_encoded = []

        for val in self.df["email"]:
            if val not in coded_dict:
                coded_dict[val] = cter
                cter += 1

            email_encoded.append(coded_dict[val])
        return email_encoded

    
    def create_user_item_matrix(self):
        '''
        Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
        an article and a 0 otherwise

        INPUT:
        df - pandas dataframe with article_id, title, user_id columns

        OUTPUT:
        user_item - user item matrix  
        '''

        user_item = self.df.groupby(["article_id", "user_id"])["title"].nunique().unstack()
        user_item.fillna(0, inplace=True)
        user_item = user_item.T
        return user_item # return the user_item matrix 

    
    def tokenize(self, text):
        """
        Tokenize, lemmatize, lower and remove punctuation of input text.

        INPUT:
            text: Single string with input text 
                  Example: 'For today:= this is, a advanced _ example #- String!'

        OUTPUT:
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
    

    def get_tokenized_articles_df(self):
        '''
        Tokenize Strings of Article content and return to seperate Dataframe.

        INPUT:
            df: Dataframe with at least the columns "doc_body", "doc_description", "doc_status", "doc_full_name"

        OUTPUT:
            df_temp: Dataframe with at least the columns "doc_full_name", "doc_description_tokens", "doc_name_tokens"
        '''

        # creating copy of df for not changing anything in the input df
        df_temp = self.df.copy()

        # drop duplicates
        df_temp = df_temp.drop_duplicates(subset=["title"])

        df_temp["title"] = df_temp["title"].str.replace("machine learning", "machinelearning")
        df_temp["title"] = df_temp["title"].str.replace("data science", "datascience")

        # tokenize title strings
        df_temp["title_tokens"] = df_temp.apply(lambda x: self.tokenize(x["title"]), axis=1)

        # remove unneccessary columns
        df_temp.drop(columns=["user_id"], inplace=True)

        return df_temp["title_tokens"]
    
    
    
if __name__ == '__main__':
    # test different parts to make sure it works
    reco = Recommender()
    print("Test Collaborative Filter based Recommendation:")
    print(reco.make_collab_recs(5))
    print("Test Content Filter based Recommendation:")
    print(reco.make_content_recs("1427.0"))
    #print(reco.fit_svd())
    #print("Get Top sorted Users")
    #print(reco.ra.get_top_sorted_users(6))
    print("Get Token Text")
    print(reco.ra.get_token_texts(['1429.0', '1330.0', '1431.0', '1427.0', '1364.0', '1314.0', '1293.0', '1170.0', '1162.0', '1304.0']))
    
    