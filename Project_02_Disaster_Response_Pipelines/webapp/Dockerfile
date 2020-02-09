FROM tiangolo/uwsgi-nginx-flask:python3.6

WORKDIR /app/

RUN pip install numpy==1.13
RUN pip install scipy==0.19.1
RUN pip install Flask==1.0.0
RUN pip install pandas==1.0.0
RUN pip install scikit_learn
RUN pip install nltk
RUN 

ENV ENVIRONMENT production

COPY main.py __init__.py /app/