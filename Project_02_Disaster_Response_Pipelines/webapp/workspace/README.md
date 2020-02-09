# Disaster Response Pipeline Project

## Landing Page with first Plotly Visualisation
![](https://imgur.com/xTMDsW5.jpg)

## Second Graph Visualisation on Landing Page
![](https://imgur.com/paJIsXT.jpg)

## Zoom into Second Graph Visualisation
![](https://imgur.com/bS8uYnF.jpg)

## Classification of Example Message "Help me. I need water."
![](https://imgur.com/95ZeX3K.jpg)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:8000/
