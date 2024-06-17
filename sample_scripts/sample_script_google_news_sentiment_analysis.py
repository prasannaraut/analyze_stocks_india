# This is a sample script which explains how to perform sentiment analysis using the google news headlines
from analyze_stocks_india.news_sentiment_analysis_Indian_Stocks import get_sentiment

data = {
    'SBIN':['state bank of india news'],
    'ONGC':['ONGC stock news'],
    'TCS': ['TCS stock news'],
    'EICHERMOT': ['Eicher Motors Ltd Stock'],
    'SUZLON': ["Suzlon energy stock"]
}

df, news = get_sentiment(data)

print(df)
#print(news)