resources = """
  * https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
  * https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
  
## Causality Resources
  * Explanation of Difference Correlation and Causalisation: https://calculatedcontent.com/2013/05/27/causation-vs-correlation-granger-causality/
  * Used Function from: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

## SARIMAX Resources

  * Good Overview about ARIMA Models: https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
  * How to prepare Explanatory Variables for Multivariate SARIMAX Model: https://www.kaggle.com/viridisquotient/arimax
  * How to prepare Time Series data for Multivariate SARIMAX Model: https://www.machinelearningplus.com/time-series/time-series-analysis-python/

"""

view_data_conclusion = """
## Conclusions from Viewing the Data

  * Gold behaves highly differently compared to Bitcoin
  * the other Stock Charts share some similarities with Bitcoin regarding peaks and bottoms --> deeper investigation
  * from the first point of view the Twitter Sentiments and Google Trends doesn't share any similarity with Bitcoin Chart --> deeper investigation
"""

granger_prob_expl = """
#### Granger Causality tests on a Null hypothesis:

  * __Null Hypothesis__: X does not granger cause Y.
  
  * If __P-Value < Significance level__ (0.05), then Null hypothesis would be rejected.
  >
  > Example: __cryptocurrency_Google_Trends_x__ does granger cause __bitcoin_Price_y__ 
  >
  
  * If __P-Value > Significance level__ (0.05), then Null hypothesis cannot be rejected.
  >  
  > Example: __alibaba_Price_x__ doesn't granger cause __gold_Price_y__
  >
"""

introduction = """
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
"""

correlation_conclusion = """
## Simple Correlation

There are some obvious correlations but some interesting as well:
  * Positive Economic Sentiments are strongly correlated with Bitcoin Indicators
  * Google Price is strongly correlated with Bitcoin Indicators
  * SP500 Price is strongly correlated with Bitcoin Indicators
  * Bitcoin Google Trends and Cryptocurrency Google Trends are strongly correlated with Bitcoin Indicators
  

## Shifting Correlation

It's very interesting, as:
  * 30 Day into Past shifted Economy Positive Sentiments increased its correlation with Bitcoin Price Indicators
  * 30 Day into Past shifted Cryptocurrency Google Trends increased its correlation with Bitcoin Price Indicators
  * 30 Day into Past shifted Bitcoin Google Trends remains its correlation with Bitcoin Price
  * 30 Day into Past shifted Google, SP500 and Dax Prices increased its correlation with Bitcoin Price
  
It's very interesting as well:
  * The Gold Price isn't correlated with Bitcoin Price in any way
"""