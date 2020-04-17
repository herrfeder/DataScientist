# Capstone Project: Multivariate Timeseries Analysis and Prediction (Stock Market)
## Purpose

Building an Time Series Forecast Application to predict and forecast __Bitcoin financial data__
using supervised and unsupervised Machine Learning Approaches, this includes:
  * search, collection and of supportive Features in form of suitable Time Series (social media, other similar charts)
  * preparation, analysis, merging of Data and Feature Engineering using:
    * Correlative Analysis
    * Stationarity Analysis
    * Causality Analysis
  * Model Preprocessing and Model Fitting with this Machine Learning Algorithms:
    * supervised SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model)
    * unsupervised GRU (Gated Recurrent Unit)
  * building an Web Application using a Dash Webapp (see folder __webapp__)
    * explains my roadmap of analysis and conclusions
    * provides feature of daily forecasting using designed models
    * Own Webapp Repository: https://github.com/herrfeder/Udacity-Data-Scientist-Capstone-Multivariate-Timeseries-Prediction-Webapp
  

## Approach/Idea

I want to find Correlation and Causality to the Bitcoin Price by shifting all other collected time series in time.
For Example: Shifting all supportive Features one month to past gives me the freedom to look one month into the future for forcasting.

Therefore the webapp will do only prediction in a timespan in the past for seeing the comparision to the true price.

## Used Data

1. Stock Market Data for the last five years from [Investing.com](https://www.investing.com) for:
  * Bitcoin
  * DAX
  * SP500
  * Google
  * Amazon
  * Alibaba
2. Google Trends for keywords "bitcoin", "cryptocurrency", "ethereum", "trading", "etf" using this notebook [00_scrape_googletrend.ipynb](blubb)
3. Twitter Sentiments for keyword "bitcoin" and "#economy" using notebooks [00_scrape_twitter.py](blubb) and [00_tweet_to_sent.ipynb](blubb)


## Used Libaries

  * Data Collection:
    * [twint](https://github.com/twintproject/twint)
    * [pytrends](https://github.com/GeneralMills/pytrends)
  * NLP:
    * [NLTK](https://github.com/nltk/nltk)
  * Webapp and Visualisation: 
    * [Plotly](https://github.com/plotly/plotly.py)
    * [dash](https://github.com/plotly/dash)
    * [matplotlib](https://github.com/matplotlib/matplotlib)
  * Data Analysis and Wrangling:
    * [Pandas](https://github.com/pandas-dev/pandas)
    * [Numpy](https://github.com/numpy/numpy)
    * [statsmodels](https://github.com/statsmodels/statsmodels)
  * Modelling and Evaluation:
    * [Numpy](https://github.com/numpy/numpy)
    * [Scikit Learn](https://github.com/scikit-learn/scikit-learn)
    * [statsmodels](https://github.com/statsmodels/statsmodels)
    * [Tensorflow](https://github.com/tensorflow/tensorflow)
    * [Keras](https://github.com/keras-team/keras)

## Screenshots of Webapp

| Forecast Application | View all Data | Granger Causality Plot |
|--------------------------------------|--------------------------------------|--------------------------------------|
| ![](https://github.com/herrfeder/Udacity-Project-Recommendations-With-IBM-Webapp/raw/8db15c2ebe164d14f956c593809874259e378a30/screenshots/recommendations_webapp_top.png) | ![](https://github.com/herrfeder/Udacity-Project-Recommendations-With-IBM-Webapp/raw/8db15c2ebe164d14f956c593809874259e378a30/screenshots/recommendations_webapp_bottom.png) | ![](https://github.com/herrfeder/Udacity-Project-Recommendations-With-IBM-Webapp/raw/8db15c2ebe164d14f956c593809874259e378a30/screenshots/recommendations_webapp_bottom.png) |

## Included Files
  
  * __webapp__: Folder that holds the files and folders for the Dash webapp. For installation and deployment, look into it
  * __00_scrape_googletrend.ipynb__: Scraping Google Trends
  * __00_scrape_twitter.py__: Scrape Twitter using Twint
  * __00_tweet_to_sent.ipynb__: Convert collected tweets to sentiment scores
  * __01_corr_analysis.ipynb__: Data Processing, Merging and Correlative Analysis
  * __02_stationarity_causality_analysis.ipynb__: Analysis for Stationarity and Causality
  * __03_1_model_ARIMAX__: Modelling and Validation for SARIMAX model
  * __03_2_model_ARIMAX_optimization.py__: Optimizing SARIMAX model by testing different sets of features
  * __03_3_investigate_feature_optimization.ipynb__: Finding model with best performance from previous test
  * __04_model_GRU.ipynb__: Modelling and Validation for GRU model
  * __data/__: Holds all source data described as above
  * __arimax_results/__: holds the results for SARIMAX feature optimization
  * __data_prep_helper.py__: consists of helper classes to read, process, shift and split data and do forecasting
  * __plot_helper.py__: consists of different supportive plotly functions 

## Brief Results

The model prediction for using "young" time series that are shifted up to a week are pretty accurate.
The model prediction for the desired month isn't far away from beeing accurate but we can see several volatile Chart Movements before they will happen and that's a nice result.

## Possible Roadmap

  * Extensive Hyperparameter Optimization: Due to a lack of time, resources and knowledge this was only done rudimentary. I'm sure the models can be improved
    by that.
  * Extend Webapp to full realtime Forecasting.
  * Check more and more different feature time series.


## Installation and Deployment

I prepared a Dockerfile that should automate the installation and deployment. 
For further instructions see folder [webapp](https://github.com/herrfeder/Udacity-Data-Scientist-Capstone-Multivariate-Timeseries-Prediction-Webapp)

I'm running the Web Application temporary on my own server: https://federland.dnshome.de/bitcoinprediction
Please be gentle, it's running on limited resources. This app __isn't responsive__.

## Acknowledgements

  * To [Nicolas Essipova](https://github.com/NicoEssi) for being my mentor for the whole Nanodegree
