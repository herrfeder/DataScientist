# Project 2: Disaster Response Pipeline Project
## Purpose
Automate and deploy the complete 
  * the ETL (Extract, Transform, Load) Pipeline
  * the following Machine Learning Pipeline
  * the resulting Application using a Flask Webapp (see folder __webapp__)
  
I prepared the webapp pipelines in the notebooks in this repository. The pre-labelled data source is originating from [Figure Eight](https://www.figure-eight.com/) that is working together with Udacity for Creation of this project.
The Data includes more than 20.000 real Text Messages that where sent during different kinds of disasters and where pre-labelled into one or multiple of 36 different categories.

Therefore the Goal is to feed the processed text messages as the features and the pre-labelled 36 Categories as the predicted output into a supervised machine learning model. 

## Used Libaries

  * Graph Visualisation: __PyVis__ and __Matplotlib__
  * Distribution Visualisation: __Plotly__
  * RandomForestClassifier and Mulitnominal Naive Bayes: __scikit learn__
  * ML Pipeline and Evaluation: __scikit learn__
  * Data Wrangling and Analysis: __pandas__
  * Webserver: __Flask__
  * Webdesign: __Bootstrap 4__ and __Bootstrap 4 Material__

## Screenshots of Webapp
## Preview

|  Landing Page with first Plotly Visualisation | Second Graph Visualisation on Landing Page |  Zoom into Second Graph Visualisation | Classification of Example Message "Help me. I need water." |
|--------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|
| ![](https://imgur.com/xTMDsW5.jpg) | ![](https://imgur.com/paJIsXT.jpg) | ![](https://imgur.com/bS8uYnF.jpg) | ![](https://imgur.com/95ZeX3K.jpg) |


## Included Files
  
  * __webapp__: Folder that holds the files and folders for the Flask webapp. For installation and deployment, look into it
  * __01_ETL_Pipeline_Preparation.ipynb__: Extract, Assess, Clean, Transform and Load Data Source into Database for ML Pipeline 
  * __02_ML_Pipeline_Preparation.ipynb__: Different Attempts of Model Fitting and Evaluation and storing the best (performance and reasonability) model 
  * __03_Testing_Visuals.ipynb__: Test Bed for creating Graph Visualisation
  * __99_test_case_classification_report.ipynb__: Experiment to draw the ideal case of an classification report
  * __categories.csv__: Data Source with 36 Categories
  * __messages.csv__: Text Messages associated with Disaster events 

## Brief Results

  * With a Random Forest Classifier and a Custom Transformer I was able to achieve a __summarized and macro averaged F1-Score around 0.80__
  * The parameter optimization with GridsearchCV seemed to be totally overfitted (see ML_Pipeline_Preparation Notebook)
    * There are some others with similiar issues but not able of resolving it:
      * https://www.reddit.com/r/datascience/comments/908vgf/why_isnt_the_best_estimator_pulled_from/
      * https://github.com/scikit-learn/scikit-learn/issues/13872
  * The deploying of the complete ETL Pipeline and Machine Learning Pipeline into a Flask Webapp was straightforward
    * the whole appearance could be improved by choosing [Dash](https://plot.ly/dash/) instead of plain Flask

## Installation and Deployment

I prepared a Dockerfile that should automate the installation and deployment.
For further instructions see folder [webapp](https://github.com/herrfeder/DataScientist/tree/master/Project_02_Disaster_Response_Pipelines/webapp/workspace)

I'm running the Web Application temporary on my own server: https://federland.dnshome.de/disasterresponse
Please be gentle, it's running on limited resources. This app __isn't responsive__.

## Acknowledgements

  * To [Nicolas](https://www.linkedin.com/in/essipova/), my Udacity Mentor for guiding me through some obstacles during this project
  * To the very good [Scikit Learn Documentation](https://scikit-learn.org/stable/), that helped me to understand a few things

