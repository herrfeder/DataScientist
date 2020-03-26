import twint
from datetime import date, timedelta
import time


def collect_daily_tweets(search):
    start_date = date(2019, 4, 25)
    end_date = date(2020, 3, 10)
    delta = timedelta(days=1)
    while start_date <= end_date:
        start_string = start_date.strftime("%Y-%m-%d") + " 00:00:00"
        end_string = start_date.strftime("%Y-%m-%d") + " 15:00:00"
        c = twint.Config()
        c.Search = search
        c.Since = start_string
        c.Until = end_string
        c.Lang = "en"
        c.Limit = 500
        c.Store_csv = True
        c.Output = search + start_string.split(" ")[0]
        
        twint.run.Search(c)
        start_date += delta
        time.sleep(30)
        
if __name__ == "__main__":

    collect_daily_tweets("#economy")