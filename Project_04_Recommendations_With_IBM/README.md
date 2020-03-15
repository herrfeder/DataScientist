# Project: Recommendations with IBM
## Purpose

Building an Recommendation Engine to recommend articles to single users based on their activity and the contents of the articles, moreover:
  * the preperation, analysis and preprocessing of the data sources 
  * designing the Recommendation Engine using Singular Value Decomposition (SVD)
  * building an Web Application using a Dash Webapp (see folder __webapp__)
    * Own Webapp Repository: https://github.com/herrfeder/Udacity-Project-Recommendations-With-IBM-Webapp.git
  
The initial analysis of the source data and evaluation can be found in this notebook [Recommendations_with_IBM.ipynb](https://github.com/herrfeder/DataScientist/blob/master/Project_04_Recommendations_With_IBM/Recommendations_with_IBM.ipynb). The pre-labelled data source is originating from [IBM Watson Studio Platform](https://dataplatform.cloud.ibm.com/) that is working together with Udacity for Creation of this project.

The Source Data includes more than 45.000 user interactions with more than 700 data science associated articles that are existing in IBM Watson Studio Platform.

Therefore __the Goal is__ to: 
  * enumerate all interactions for any user and make them comparable to each other
  * finding ways of measuring the similarity between users and articles
  * make the measurements and recommendation usable for the customer

## Used Libaries

  * Webapp and Visualisation: __[Plotly](https://github.com/plotly/plotly.py)__ and __[dash](https://github.com/plotly/dash)__
  * Data Analysis and Wrangling: __[Pandas](https://github.com/pandas-dev/pandas)__
  * Recommendation and SVD: __[Scipy](https://github.com/scipy/scipy)__ and __[Numpy](https://github.com/numpy/numpy)__
  * Wordcloud: __[word_cloud](https://github.com/amueller/word_cloud)__

## Screenshots of Webapp

| Control Panel and Recommendation Area | Wordcloud and Most Similiar Users |
|--------------------------------------|--------------------------------------|
| ![](screenshots/recommendations_webapp_top.png) | ![](screenshots/recommendations_webapp_bottom.png) |

## Included Files
  
  * __webapp__: Folder that holds the files and folders for the Dash webapp. For installation and deployment, look into it
  * __Recommendations_with_IBM.ipynb__: Notebook that holds complete process of Data Wrangling, Analysis and Recommendation Modelling and Evaluation
  * __Recommendations_with_IBM.html__: HTML represenation of this notebook for easier access
  * __project_tests.py__: Provided by Udacity for evaluating my results
  * __data/articles_community.csv__: All articles and complete content of them
  *  __data/user-item-interactions.csv__: Source data with all user interactions for all articles
  * __data/user_item_matrix.p__: 2-dimensional DataFrame with Articles in columns and user id's as index

## Brief Results

  * The SVD based recommendation engine doesn't perform well because of a high sparsity in the input matrix
    * The input matrix holds binary values (0 for False, 1 for True) and has therefore many zeroes in it
    * Therefore the increasing of latent factors leads fast to overfitting
  * Using intersections of NLP tokenized string groups is a good basis for a content based recommendation
  * By converting the notebook functions into classes the creation of the Dash Webapp was straightforward

## Installation and Deployment

I prepared a Dockerfile that should automate the installation and deployment. 
For further instructions see folder [webapp](https://github.com/herrfeder/DataScientist/tree/master/Project_02_Disaster_Response_Pipelines/webapp/workspace)

I'm running the Web Application temporary on my own server: https://federland.dnshome.de/recommendations
Please be gentle, it's running on limited resources. This app __isn't responsive__.

## Acknowledgements

  * To [Prashant Saikia](https://github.com/PrashantSaikia) for providing such a good repo on creating wordclouds with plotly
