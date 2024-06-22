#source: https://medium.com/@wisjnujudho/how-to-scrape-google-news-top-stories-bs4-nopagination-80b882a214e5

import pandas as pd
from bs4 import BeautifulSoup
import requests

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import analyze_stocks_india.supporting_functions as sf

pd.set_option('display.max_columns', 12)


#symbols = ["SBIN","IDEA"]
#news_url = ["https://news.google.com/search?q=state+bank+of+india+news+when:7d","https://news.google.com/search?q=idea+vodafone+news+when:7d"]

def get_sentiment(data, days=7):
    '''
    :param symbols: python dictionary with key as symbols and value as keywords to search news
    :param days: days to consider for news articles from past (in days)
    :return: touple (df, news). df is a pandas dataframe containing sentiment values from -1 to 1. news contains the news headlines considered for the sentiment analysis
    '''

    symbols=list(data.keys())
    #symbols=["SBIN", "CIPLA", "BAJFINANCE", "ONGC"]

    news_url = []
    for s in symbols:
        # print(data[s])
        news_url.append(sf.get_google_news_url_v2(data[s],days))
    # news_url = [get_google_news_url(["state bank of india news"],7),
    #             get_google_news_url(["Cipla company news"],7),
    #             get_google_news_url(["Bajaj Finance news"],7),
    #             get_google_news_url(["ONGC News"],7)
    #             ]
    df, news = sf.get_news_sentimet_v2(symbols, news_url)
    #print(df)
    #print(news)
    return (df, news)

