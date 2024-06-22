import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import analyze_stocks_india.supporting_functions as sf
import pathlib
import os



############################################## Parameters Update Here #####################################################

symbol = "SBIN"

############################################## Parameters Update End Here #####################################################



def create_plot_trend(data,symbol):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["CLOSE"],
        mode = "lines",
        name = "CLOSE"
    ))

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["HIGH"],
        mode="lines",
        name="HIGH"
    ))

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["LOW"],
        mode="lines",
        name="LOW"
    ))

    fig.update_layout(
        title="Close price with High and Low for - {}".format(symbol),
        plot_bgcolor='white',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
    )

    #fig.show()
    return fig


def create_plot_quantity(data,symbol):
    fig = make_subplots(rows=3,cols=1)

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["TOTTRDQTY"],
        mode = "lines",
        name = "TOTTRDQTY"),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["TOTTRDVAL"],
        mode="lines",
        name="TOTTRDVAL",),
        row=2, col=1
    )

    fig.add_trace(go.Scatter(
        x=data["Date"], y=data["TOTALTRADES"],
        mode="lines",
        name="TOTALTRADES",),
        row=3, col=1
    )

    fig.update_layout(
        title="Traded Quantity/Value/Trades - {}".format(symbol),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )



    return fig


def get_trend_plot(symbol):
    '''
    :param symbol: nse symbol eg "SBIN"
    :return: fig class
    '''
    try:
        dir_path = pathlib.Path(__file__).parents[0]
        dir_data_files = str(dir_path)+'\\data_files\\processed\\'

        data = pd.read_csv(dir_data_files + "{}.csv".format(symbol))
        columns=data.columns #['Date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
        data = data[columns]
        data['Date'] = pd.to_datetime(data['Date'].apply(lambda x: x.split()[0]))
        fig = create_plot_trend(data,symbol)
        #fig.show()
        return fig

    except:
        print("Something happend \n Maybe the stock that you are looking for is not present in data_files --> processed directory \n please download and process the stock data")

def get_trade_quantity(symbol):
    '''
    :param symbol: nse symbol eg "SBIN"
    :return: fig class
    '''
    try:
        dir_path = pathlib.Path(__file__).parents[0]
        dir_data_files = str(dir_path) + '\\data_files\\processed\\'

        data = pd.read_csv(dir_data_files + "{}.csv".format(symbol))
        columns = data.columns  # ['Date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
        data = data[columns]
        data['Date'] = pd.to_datetime(data['Date'].apply(lambda x: x.split()[0]))
        fig = create_plot_quantity(data, symbol)
        #fig.show()
        return fig

    except:
        print("Something happend \n Maybe the stock that you are looking for is not present in data_files --> processed directory \n please download and process the stock data")


'''
symbols = sf.ipos_2023_2022_2021

for symbol in symbols:

    data=pd.read_csv('processed_data\\{}.csv'.format(symbol))

    columns=data.columns #['Date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
    data = data[columns]

    data['Date'] = pd.to_datetime(data['Date'].apply(lambda x: x.split()[0]))
    #data.set_index('Date',drop=True,inplace=True)

    fig = create_plot_trend(data,symbol)
    fig.show()

    #fig = create_plot_quantity(data,symbol)
    #fig.show()
'''