# This file is to keep all the supporting functions
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import pathlib

from bs4 import BeautifulSoup
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Union, Optional, List, Dict, Tuple, TextIO, Any


'''
Get moving average, inputs are dataframe, column for which moving average is required, days for average
Returns series of moving average
'''
def get_moving_avg_bollinger_bands(df,col,days):
    #get required column as series
    data = df[col]

    #defing window
    windows = data.rolling(days)

    #get moving average
    moving_avg = windows.mean()
    std = windows.std()
    upper_bollinger = moving_avg + 2*std
    lower_bollinger = moving_avg - 2*std

    return [moving_avg,upper_bollinger, lower_bollinger]



'''
Loads data from extracted csv files as pandas dataframe
'''
def load_data(symbol):
    dir_path = pathlib.Path(__file__).parents[0]
    dir_data_files = str(dir_path) + '\\data_files\\processed\\'

    data = pd.read_csv(dir_data_files + '{}.csv'.format(symbol))
    columns = data.columns  # ['Date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
    data = data[columns]
    return data


'''
Creates plots for multiple columns in the dataset, optionally title can be supplied to the function
Returns fig 
'''
def create_plot_from_symbol(s,col_list, title="Title"):
    data = load_data(s)
    fig = go.Figure()

    for col in col_list:

        fig.add_trace(go.Scatter(
            x=data["Date"], y=data[col],
            mode = "lines",
            name = col
        ))

    fig.update_layout(
        title=title,
        plot_bgcolor='white',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
    )

    #fig.show()
    return fig


'''
Check if mean reversion can be used for the stock.
A key characteristic of the Mean Reversion strategy is that it profits mostly in sideways markets and loses mostly in trending markets.
Returns true or false
'''

def check_mean_reversion(symbol,mv_days=20):
    data = load_data(symbol)

    data["moving_average"] = get_moving_avg_bollinger_bands(data, "CLOSE", mv_days)[0]
    data['priceBymv'] = data["CLOSE"] / data["moving_average"]
    mean = data['priceBymv'].mean()
    if (mean > 0.99) and (mean < 1.01):
        return True
    else:
        return False


'''
get RSI (https://en.wikipedia.org/wiki/Relative_strength_index)
The relative strength index (RSI) is a technical indicator used in the analysis of financial markets. 
It is intended to chart the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period. 

The RSI is most typically used on a 14-day timeframe, (here using 7-day timeframe)
measured on a scale from 0 to 100, with high and low levels marked at 70 and 30, respectively. 
Short or longer timeframes are used for alternately shorter or longer outlooks. 

High and low levels—80 and 20, or 90 and 10—occur less frequently but indicate stronger momentum. 
'''
def get_RSI(data,period=7):

    def gain(value):
        if value < 0:
            return 0
        else:
            return value

    def loss(value):
        if value > 0:
            return 0
        else:
            return abs(value)


    # Calculate price delta, gain and loss
    data['delta'] = data["CLOSE"].diff()
    data['gain'] = data["delta"].apply(lambda x: gain(x))
    data['loss'] = data["delta"].apply(lambda x: loss(x))

    # calculate ema (exponential moving average)
    data['ema_gain'] = data['gain'].ewm(period).mean()
    data['ema_loss'] = data['loss'].ewm(period).mean()

    # calculate RSI
    data["rs"] = data['ema_gain'] / data['ema_loss']
    RSI = data['rs'].apply(lambda x: 100 - (100 / (x + 1)))

    return RSI

'''
Predicts the buy or sell indicator based on RSI value. 
Input as dataset, optionally low and up bound for RSI can be specified.
Alpha, can be provided to relax buy signal. if alpha is 0, no relaxation, if alpha is 0.05 then 5% relaxation. Ideally 0.5% relaxation is good to predict buy signal 1 day prior
Returns the dataset with Closing value and Indication for each day and figure

how to use
rsi_data, fig = sf.predict_buy_sell_from_RSI(data,25,75)
'''
def predict_buy_sell_from_RSI(data, low_bound=30, up_bound=70, alpha = 0.05):
    data["RSI"] = get_RSI(data, 7)

    [mv, upper_bollinger, lower_bollinger] = get_moving_avg_bollinger_bands(data, "CLOSE", 20)
    data["moving_avg"] = mv
    data["upper_bollinger"] = upper_bollinger
    data["lower_bollinger"] = lower_bollinger

    data['priceBymv'] = data["CLOSE"] / data["moving_avg"]

    # Buy Signal (1)
    data['RSI_Signal'] = np.where(
        (data['RSI'] < low_bound * (1+alpha)) &
        (data['CLOSE'] < data['lower_bollinger'] * (1+alpha)), 1, np.nan)

    # Sell Signal (-1)
    data['RSI_Signal'] = np.where(
        (data['RSI'] > up_bound) &
        (data['CLOSE'] > data['upper_bollinger']), -1, data['RSI_Signal'])

    # buy/sell signal for next trading day
    data["RSI_Signal"] = data["RSI_Signal"].fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["CLOSE"],
        mode="lines",
        line=dict(color='black', dash='solid'),
        name="Day Closing Price"
    ))

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["moving_avg"],
        mode="lines",
        line=dict(color='black', dash='dash'),
        name="20 day moving avg"
    ))

    fig.add_trace(go.Scatter(
        x=data[(data["RSI_Signal"] == 1)]["Date"], y=data[(data["RSI_Signal"] == 1)]["CLOSE"],
        mode="markers",
        marker={'color': 'green', 'size': 10},
        name="Buy"
    ))

    fig.add_trace(go.Scatter(
        x=data[(data["RSI_Signal"] == -1)]["Date"], y=data[(data["RSI_Signal"] == -1)]["CLOSE"],
        mode="markers",
        marker={'color': 'red', 'size': 10},
        name="Sell"
    ))

    fig.update_layout(
        #title="Close price with High and Low for - {}".format(symbol),
        plot_bgcolor='white',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
    )

    return (data[["Date","moving_avg", "CLOSE", "RSI", "RSI_Signal"]], fig)



'''
Provide a list of symbols (companies)
check for each company if the buy signal was predicted in last n days.
if yes provide the list of that stock data and figures

#how to use
rsi_data_list, fig_list = sf.predict_stock_to_buy_using_RSI(sf.nifty_100,7,30,70)
for fig in fig_list:
    fig.show()

'''
def predict_stock_to_buy_using_RSI(symbols, n_days = 3, rsi_low_bound=30, rsi_up_bound=70, alpha=0.05):
    rsi_data_list = []
    fig_list = []
    mean_reversion_list = []
    symbols_buy = []

    for s in symbols:
        print("Checkign for: {}".format(s))
        data = load_data(s)
        rsi_data, fig = predict_buy_sell_from_RSI(data, rsi_low_bound, rsi_up_bound, alpha)

        condition = 1 in (rsi_data["RSI_Signal"][n_days * -1:]).unique()  # checks if buy signal was triggerd in last 3 days

        if (condition):
            rsi_data_list.append(rsi_data)

            fig.update_layout(
                title="for {} with Mean Reversion = {}".format(s, check_mean_reversion(s)))
            fig_list.append(fig)

            mean_reversion_list.append(check_mean_reversion(s))

            symbols_buy.append(s)

    return [symbols_buy, rsi_data_list, fig_list, mean_reversion_list]

'''
Get news sentiment for a particular company
input: symbol and url (note both should be a list)
output: news list of headline, date, sentiment score
'''
def get_news_sentimet(symbols,urls):
    tickers = symbols
    news_urls = urls

    all_news = []
    for i in range(len(tickers)):
        ticker = tickers[i]
        news_url = news_urls[i]
        page = requests.get(news_url).text
        soup = BeautifulSoup(page, 'html.parser')

        #  2. PARSING HTML
        # title
        result_tl = soup.select('article .DY5T1d.RZIKme')
        title = [t.text for t in result_tl]

        # date-time
        result_dt = soup.select('[datetime]')
        timedate = [d['datetime'] for d in result_dt]

        ticker_list_ = [ticker for t in result_tl]

        all_data = list(zip(ticker_list_, timedate, title))
        all_news.extend(all_data)

    # print(all_news)
    # convert the list into dataframe
    parsed_news = pd.DataFrame(all_news, columns=['Ticker', 'Date', 'Headline'])

    # Sentiment Analysis
    # Using the powerful nltk module, each headline is analyzed for its polarity score on a scale of -1 to 1, with -1 being highly negative and highly 1 being positive.
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    # View Data
    # print(news)
    # news['Date'] = pd.to_datetime(news.Date).dt.date

    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

    values = []
    for ticker in tickers:
        dataframe = news_dict[ticker]
        dataframe = dataframe.set_index('Ticker')
        dataframe = dataframe.drop(columns=['Headline'])
        #print('\n')
        # print(dataframe.head())
        print("{} news from {}".format(dataframe.shape[0], ticker))

        mean = round(dataframe['compound'].mean(), 2)
        values.append(mean)

    df = pd.DataFrame(list(zip(tickers, values)), columns=['Ticker', 'Mean Sentiment'])
    df = df.set_index('Ticker')
    df = df.sort_values('Mean Sentiment', ascending=False)
    # print('\n')
    # print(df)

    return (df, news)



'''
To generate goole search url from keywords
input: list of keywords, number of 7 to chekck in past for news
output: url link
'''
def get_google_news_url(keywords, days):
    # sample
    # "https://news.google.com/search?q=state+bank+of+india+news+when:7d","https://news.google.com/search?q=idea+vodafone+news+when:7d"
    url = "https://news.google.com/search?q="
    for k in keywords:
        url += k
        url += "+"
    url += "when:"
    url += str(days)
    url += "d"
    return url


def get_news_sentimet_v2(symbols,urls):
    tickers = symbols
    news_urls = urls

    # a python list which contains news articles for each ticker
    news_titles = []
    for i in range(len(tickers)):
        ticker = tickers[i]
        url = news_urls[i]

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        articles = soup.find_all('article')
        links = [article.find('a')['href'] for article in articles]
        links = [link.replace("./articles/", "https://news.google.com/articles/") for link in links]

        news_text = [article.get_text(separator='\n') for article in articles]
        news_text_split = [text.split('\n') for text in news_text]

        news_df = pd.DataFrame({
            'Title': [text[2] for text in news_text_split],
            'Source': [text[0] for text in news_text_split],
            'Time': [text[3] if len(text) > 3 else 'Missing' for text in news_text_split],
            'Author': [text[4].split('By ')[-1] if len(text) > 4 else 'Missing' for text in news_text_split],
            'Link': links
        })
        news_titles.append(news_df['Title'])


    # Sentiment Analysis
    # Using the powerful nltk module, each headline is analyzed for its polarity score on a scale of -1 to 1, with -1 being highly negative and highly 1 being positive.

    mean_sentiment = []
    for i in range(len(tickers)):
        analyzer = SentimentIntensityAnalyzer()
        t_news = list(news_titles[i])

        ticker_scores_list = []
        for n in t_news:
            ticker_scores_list.append(analyzer.polarity_scores(n)['compound'])

        mean = sum(ticker_scores_list)/len(ticker_scores_list)
        mean_sentiment.append(mean)

    df = pd.DataFrame(list(zip(tickers, mean_sentiment)), columns=['Ticker', 'Mean Sentiment [-1,1]'])
    df = df.set_index('Ticker')
    df = df.sort_values('Mean Sentiment [-1,1]', ascending=False)
    # print('\n')
    # print(df)

    return (df, news_titles)


'''
To generate goole search url from keywords
input: list of keywords, number of 7 to chekck in past for news
output: url link
'''
def get_google_news_url_v2(keywords, days):
    # sample
    # "https://news.google.com/search?q=state+bank+of+india+news+when:7d","https://news.google.com/search?q=idea+vodafone+news+when:7d"
    url = "https://news.google.com/search?q="
    for k in keywords:
        url += k
        url += "+"
    url += "when:"
    url += str(days)
    url += "d"
    url += "&hl=en-IN&gl=IN&ceid=IN:en"
    return url


'''
To get stock information from NSE website as JSON data
Input: Stock Symbol
Output: Stock info in JSON Format
'''
def get_stock_info_json(stock):
    url_stock: str = "https://www.nseindia.com/api/option-chain-equities?symbol="  #"https://www.nseindia.com/api/option-chain-equities?symbol="
    url_oc: str = "https://www.nseindia.com/option-chain"

    url = url_stock+stock

    headers: Dict[str, str] = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                      'like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        'accept-language': 'en,gu;q=0.9,hi;q=0.8',
        'accept-encoding': 'gzip, deflate, br'}



    session: requests.Session = requests.Session()
    response = session.get(url, headers=headers, timeout=10)

    if response.status_code == 401:
        session.close()
        session: requests.Session = requests.Session()
        request = session.get(url_oc, headers=headers, timeout=10)
        cookies = dict(request.cookies)
        response = session.get(url, headers=headers, timeout=10, cookies=cookies)


    json_data: Dict[str, Any]

    if response is not None:
        try:
            json_data = response.json()
            #print(response)
        except Exception as err:
            #print(response)
            #print(err, sys.exc_info()[0], "2")
            json_data = {}

    session.close()
    return json_data







'''
To get stock information from NSE website as JSON data
Input: INDEX Symbol
Output: INDEX info in JSON Format
'''
def get_index_info_json(index):
    url_index: str = "https://www.nseindia.com/api/option-chain-indices?symbol="
    url_oc: str = "https://www.nseindia.com/option-chain"
    url = url_index+index

    headers: Dict[str, str] = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                      'like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        'accept-language': 'en,gu;q=0.9,hi;q=0.8',
        'accept-encoding': 'gzip, deflate, br'}

    session: requests.Session = requests.Session()
    response = session.get(url, headers=headers, timeout=5)
    #print(response)

    if response.status_code == 401:
        session.close()
        session: requests.Session = requests.Session()
        request = session.get(url_oc, headers=headers, timeout=5)
        cookies = dict(request.cookies)
        response = session.get(url, headers=headers, timeout=10, cookies=cookies)
        #print(response)


    json_data: Dict[str, Any]

    if response is not None:
        try:
            json_data = response.json()
            #print(response)
        except Exception as err:
            #print(response)
            #print(err, sys.exc_info()[0], "2")
            json_data = {}

    session.close()
    return json_data



'''
get nearest 3 strick prices from option chain data
input: jason format of option chain of a stock
output: list of 3 strick prices which are nearest
'''
def get_3_nearest_strickPrices(stock):
    def get_expiryDates(stock):
        return stock['records']['expiryDates']

    def get_timestamp(stock):
        return stock['records']['timestamp']

    def get_value(stock):
        return stock['records']['underlyingValue']

    def get_strickPrices(stock):
        return stock['records']['strikePrices']

    sp = get_strickPrices(stock)
    value = get_value(stock)

    delta = sp[1] - sp[0]

    index_nearest_strikePrice = round((value - sp[0]) / delta)

    return [sp[index_nearest_strikePrice - 1], sp[index_nearest_strikePrice], sp[index_nearest_strikePrice + 1]]



'''
get the averageOpenInterst_percent, avg_priceChange_percent, avg_IV, avg_openInterst of call/put options of the stock for stock, taking nearest 3 strick prices of current price.
This will provide the immediate movement of the stock.
negative averageOpenInterst_percent from call options means bullish nature of the stock
positive averageOpenInterst_percent from put options means bullish nature of the stock
Input: stock_json string from get_index_info_json function
output: list of [avg_open_interst_change_percent, avg_percent_LTP_change, avg_open_interest_change, avg_IV]
'''
def get_nearest3_avgOpenInterest(stock,option):
    # data of each open interst entry
    # {'strikePrice': 560, 'expiryDate': '30-Nov-2023', 'underlying': 'SBIN', 'identifier': 'OPTSTKSBIN30-11-2023CE560.00',
    # 'openInterest': 2535, 'changeinOpenInterest': -500, 'pchangeinOpenInterest': -16.474464579901152,
    # 'totalTradedVolume': 6697, 'impliedVolatility': 11.44, 'lastPrice': 4.15, 'change': -1.4499999999999993, 'pChange': -25.892857142857135,
    # 'totalBuyQuantity': 1153500, 'totalSellQuantity': 777000, 'bidQty': 6000, 'bidprice': 4.1, 'askQty': 10500, 'askPrice': 4.2, 'underlyingValue': 560.7}

    PEorCE = ""
    if option == 'call_option':
        PEorCE = 'CE'
    elif option == 'put_option':
        PEorCE = 'PE'
    else:
        print("Please specify call_option or put_option")




    # et global values
    nearest_strickPrices = get_3_nearest_strickPrices(stock)
    nearest_expiry = stock['records']['expiryDates'][0]
    all_data = stock['records']["data"]

    # initialize
    total_open_interest_change = 0
    total_open_interst_change_percent = 0
    total_IV = 0
    total_percent_LTP_change = 0

    # calculate total
    count = 0
    for d in all_data:
        if (d['strikePrice'] in nearest_strickPrices) and (d['expiryDate'] == nearest_expiry):
            count += 1
            if(PEorCE in d.keys()):
                total_open_interest_change += d[PEorCE]['openInterest']
                total_open_interst_change_percent += d[PEorCE]['pchangeinOpenInterest']
                total_IV += d[PEorCE]['impliedVolatility']
                total_percent_LTP_change += d[PEorCE]['pChange']
            else:
                total_open_interest_change += 0
                total_open_interst_change_percent += 0
                total_IV += 0
                total_percent_LTP_change += 0

    # calculate average
    avg_open_interest_change = total_open_interest_change / count
    avg_open_interst_change_percent = total_open_interst_change_percent / count
    avg_IV =  total_IV / count
    avg_percent_LTP_change = total_percent_LTP_change / count

    return [avg_open_interst_change_percent, avg_percent_LTP_change, avg_open_interest_change, avg_IV]



'''
Predict immediate moment of stock
Input: stock_json string from get_index_info_json function
Oputput: Bearish or Bullish or Neutral
'''
def get_immediate_moment(stock):
    call_option_info = get_nearest3_avgOpenInterest(stock,'call_option')
    put_option_info = get_nearest3_avgOpenInterest(stock,'put_option')

    output = 'Neutral'
    output += " [CE: OPc%={}  LTPc%={}     PE: OPc%={}  LTPc%={}]".format(round(call_option_info[0],2),  round(call_option_info[1],2), round(put_option_info[0],2), round(put_option_info[1],2))

    if call_option_info[0]<0 and put_option_info[0]>0: #for call options openInterst reducing and for put options openInterst increasing
        output = "Bullish"
        output += " [CE: OPc%={}  LTPc%={}     PE: OPc%={}  LTPc%={}]".format(round(call_option_info[0],2),  round(call_option_info[1],2), round(put_option_info[0],2), round(put_option_info[1],2))

    elif call_option_info[0]>0 and put_option_info[0]<0: #for call options openInterst increasing and for put options openInterst reducing
        output = "Bearish"
        output += " [CE: OPc%={}  LTPc%={}     PE: OPc%={}  LTPc%={}]".format(round(call_option_info[0],2),  round(call_option_info[1],2), round(put_option_info[0],2), round(put_option_info[1],2))

    return output




'''
Get OpenInterst, IV,LTP_changePercent and strike price for PE and CE options as pandas dataframe
Input: stock_json string from get_index_info_json function and expiry [#expiry 1: this month, 2: next month, 3: month after next]
Output: pandas dataframe as output
'''
def get_OpenIntest_IV_for_allStrikePrice(stock, expiry=1): #expiry 1: this month, 2: next month, 3: month after next
    expiry_date = stock['records']['expiryDates'][expiry-1]
    all_data = stock['records']["data"]

    # initialize
    open_interest_CE = []
    open_interest_Pchange_CE = []
    IV_CE = []
    LTP_Pchange_CE = []

    open_interest_PE = []
    open_interest_Pchange_PE = []
    IV_PE = []
    LTP_Pchange_PE = []

    strike_price = []
    value = []

    # get all required data
    for d in all_data:
        if (d['expiryDate'] == expiry_date):

            if('CE' in d.keys()):
                open_interest_CE.append(d["CE"]['openInterest'])
                open_interest_Pchange_CE.append(d["CE"]['pchangeinOpenInterest'])
                IV_CE.append(d["CE"]["impliedVolatility"])
                LTP_Pchange_CE.append(d["CE"]['pChange'])
            else: #if CE data is not present
                open_interest_CE.append(0)
                open_interest_Pchange_CE.append(0)
                IV_CE.append(0)
                LTP_Pchange_CE.append(0)

            if ('PE' in d.keys()):
                open_interest_PE.append(d["PE"]['openInterest'])
                open_interest_Pchange_PE.append(d["PE"]['pchangeinOpenInterest'])
                IV_PE.append(d["PE"]["impliedVolatility"])
                LTP_Pchange_PE.append(d["PE"]['pChange'])
            else: #if PE data is not present
                open_interest_PE.append(0)
                open_interest_Pchange_PE.append(0)
                IV_PE.append(0)
                LTP_Pchange_PE.append(0)

            strike_price.append(d["strikePrice"])
            value.append(stock['records']['underlyingValue'])

    df = pd.DataFrame(data = list(zip(value, strike_price, open_interest_CE, open_interest_Pchange_CE, IV_CE, LTP_Pchange_CE, open_interest_PE, open_interest_Pchange_PE, IV_PE, LTP_Pchange_PE)),
                      columns=["current_value", "strike_price", "open_interest_CE", "open_interest_Pchange_CE", "IV_CE", "LTP_Pchange_CE", "open_interest_PE", "open_interest_Pchange_PE", "IV_PE", "LTP_Pchange_PE"])


    return df



'''
Get 3 resistance levels with Percent change in Open interst, Percent Change in LTP and IV
Input: stock_json string from get_index_info_json function and expiry [#expiry 1: this month, 2: next month, 3: month after next]
Output: strike priecs and coressponding other data
'''
def get_resistance_bullish(stock,expiry=1):
    data = get_OpenIntest_IV_for_allStrikePrice(stock, 1) #get all data
    data = data[data["open_interest_Pchange_CE"]<0] #filter when open interest change is negative (signifying bullish nature)

    data_CE = data.sort_values(by="open_interest_CE", ascending=False).head(3)
    return data_CE[["current_value", "strike_price", "open_interest_CE", "open_interest_Pchange_CE", "LTP_Pchange_CE", "IV_CE"]]



'''
Get 3 resistance levels with Percent change in Open interst, Percent Change in LTP and IV
Input: stock_json string from get_index_info_json function and expiry [#expiry 1: this month, 2: next month, 3: month after next]
Output: strike priecs and coressponding other data
'''
def get_resistance_bearish(stock,expiry=1):
    data = get_OpenIntest_IV_for_allStrikePrice(stock, 1) #get all data
    data = data[data["open_interest_Pchange_CE"]>0] #filter when open interest change is positive (signifying bearish nature)

    data_CE = data.sort_values(by="open_interest_CE", ascending=False).head(3)
    return data_CE[["current_value", "strike_price", "open_interest_CE", "open_interest_Pchange_CE", "LTP_Pchange_CE", "IV_CE"]]


'''
Get 3 supports levels with Percent change in Open interst, Percent Change in LTP and IV
Input: stock_json string from get_index_info_json function and expiry [#expiry 1: this month, 2: next month, 3: month after next]
Output: strike priecs and coressponding other data
'''
def get_support_bullish(stock,expiry=1):
    data = get_OpenIntest_IV_for_allStrikePrice(stock, 1)
    data = data[data["open_interest_Pchange_PE"] > 0]  # filter when open interest change is positive (signifying bullish nature)

    data_CE = data.sort_values(by="open_interest_PE", ascending=False).head(3)
    return data_CE[["current_value", "strike_price", "open_interest_PE", "open_interest_Pchange_PE", "LTP_Pchange_PE", "IV_PE"]]


'''
Get 3 supports levels with Percent change in Open interst, Percent Change in LTP and IV
Input: stock_json string from get_index_info_json function and expiry [#expiry 1: this month, 2: next month, 3: month after next]
Output: strike priecs and coressponding other data
'''
def get_support_bearish(stock,expiry=1):
    data = get_OpenIntest_IV_for_allStrikePrice(stock, 1)
    data = data[data["open_interest_Pchange_PE"] < 0]  # filter when open interest change is negative (signifying bearish nature)

    data_CE = data.sort_values(by="open_interest_PE", ascending=False).head(3)
    return data_CE[["current_value", "strike_price", "open_interest_PE", "open_interest_Pchange_PE", "LTP_Pchange_PE", "IV_PE"]]


'''
Get stock_moment by expiry date
Input: stock_json string from get_index_info_json function and expiry [#expiry 1: this month, 2: next month, 3: month after next]
Output: Get Bullish or Bearish, percent_chance_of_moment, avg_LTP_PercentChange, Supports and Resistances
Output: [movement, movement_percent, avg_LTP_pChange, supports, resistances, current_value]
'''
def get_expiryDate_moment(stock,expiry=1):
    support_info_bullish = get_support_bullish(stock,1)
    support_info_bearish = get_support_bearish(stock, 1)
    resistance_info_bullish = get_resistance_bullish(stock, 1)
    resistance_info_bearish = get_resistance_bearish(stock, 1)

    support_info_bullish["delta"]=support_info_bullish["current_value"]-support_info_bullish["strike_price"]
    support_info_bearish["delta"] = support_info_bearish["current_value"] - support_info_bearish["strike_price"]
    resistance_info_bullish["delta"] = resistance_info_bullish["strike_price"] - resistance_info_bullish["current_value"]
    resistance_info_bearish["delta"] = resistance_info_bearish["strike_price"] - resistance_info_bearish["current_value"]

    weighted_support_OI_bullish = np.mean([np.dot(x,y) for x, y  in zip(support_info_bullish.delta, support_info_bullish.open_interest_PE)])
    weighted_support_OI_bearish= np.mean([np.dot(x, y) for x, y in zip(support_info_bearish.delta, support_info_bearish.open_interest_PE)])
    weighted_resistance_OI_bullish = np.mean([np.dot(x, y) for x, y in zip(resistance_info_bullish.delta, resistance_info_bullish.open_interest_CE)])
    weighted_resistance_OI_bearish = np.mean([np.dot(x, y) for x, y in zip(resistance_info_bearish.delta, resistance_info_bearish.open_interest_CE)])

    weighted_bullish_OI = weighted_support_OI_bullish + weighted_resistance_OI_bullish
    weighted_bearish_OI = weighted_support_OI_bearish + weighted_resistance_OI_bearish

    movement = 'Neutral'
    movement_percent = 0
    avg_LTP_pChange = 0
    if weighted_bearish_OI < weighted_bullish_OI:
        movement = "Bullish"
        movement_percent = round(weighted_bullish_OI*100/(weighted_bearish_OI+weighted_bullish_OI),2)
        avg_LTP_pChange = resistance_info_bullish["LTP_Pchange_CE"].mean()

    if weighted_bearish_OI > weighted_bullish_OI:
        movement = "Bearish"
        movement_percent = round(weighted_bearish_OI*100/(weighted_bearish_OI+weighted_bullish_OI),2)
        avg_LTP_pChange = support_info_bearish["LTP_Pchange_PE"].mean()


    supports = list(support_info_bearish['strike_price'])
    resistances = list(resistance_info_bullish['strike_price'])
    current_value = list(resistance_info_bullish['current_value'])[0]

    return [movement, movement_percent, round(avg_LTP_pChange,2), supports, resistances, current_value]