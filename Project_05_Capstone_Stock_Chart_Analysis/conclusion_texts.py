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
# Initial Data Preprocessing and Correlative Analysis

This notebook will 

## Take Data
  * read the crypto chart data and stock chart data from [Investing.com](https://www.investing.com/) for last five years
  * read the collected Google Trend values for last five years 
  * read the collected daily twitter sentiments for search terms "bitcoin" and "#economy"

## Preprocess Data
  * convert string representations of numbers to float
  * convert human readable presentations of numbers to float
  * apply daily datetime as index for all imported dataframes
  
## Financial Analysis
  * apply moving average, moving standard deviation and Bollinger Bands for main asset Bitcoin Price
  
## Merge Data
  * merge all imported dataframes into one single dataframe
  
## Correlative Analysis
  * compute correlation matrix for all included time series
  * compute correlation matrix for static shifted time series
  * compute correlation for continuous shifting of bitcoin and stock market charts to all other indicators
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