# Project 01: Write a Data Science Blog Post
## Purpose
This project has the purpose to familiarize myself with the Data Science Process according to CRISP-DM (Cross Industry Standard Process for Data Mining). Therefore I have choosen a AirBnB Dataset from Kaggle. This dataset was scraped for over an year and built on the 07th of November 2019 by Murray Cox. He runs a website https://insideairbnb.com/ where he provides similiar and updated datasets for several big cities that have many AirBnB accomodations. His project aims at demonstrating the bad influence of AirBnB to the housing market. It's definitly worth a visit.

## Used Libaries

  * Geospatial visualization: __plotly__
  * Geospatial Transformation: __pyproj__
  * Heatmap visualizations: __seaborn__
  * Linear Model and KNN: __scikit learn__
  * Data Wrangling and Analysis: __pandas__
  
## Included Files
  
  * __Project_01_Notebook.ipynb__: Notebook that holds complete analysis
  * __listings.tar.gz__: Compressed csv, that holds information about all Berlin AirBnB accomodations
  * __listings_summary.tar.gz__: Compressed csv, that holds very detailed information about all Berlin AirBnB accomodations
  * __calendar_summary.tar.gz__: Compressed csv, that holds the daily availability and price of all Berlin AirBnB accomodations between November 2018 and November 2019
  * __neighbourhoods.tar.gz__: Compressed csv, that holds a mapping from street names to the district

## Brief Results

  * There is a relevant relation between location and the price of an AirBnB accomodation.
  * AirBnB accomodations have the highest availability during the summer.
  * it wasn't possible to apply a reasonable KNN-based learning model using the coordinates only as input data and the categorized prices as result vector

## Acknowledgements

  * To Murray Cox for the provided data from https://insideairbnb.com/
  * To XChiron for providing good tutorial for applying KNN on geospatial data: https://www.kaggle.com/xchiron/rental-list-knn-on-lat-long-data

Moreover I released my findings in a Medium post: https://medium.com/@d.lassig/how-you-will-find-the-best-price-and-availability-of-berlins-airbnb-offers-a6c8f8d2382e