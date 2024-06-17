# source: https://www.linkedin.com/pulse/algorithmic-trading-mean-reversion-using-python-bryan-chen
'''
The Mean Reversion strategy assumes that the price of a stock will eventually revert to their long-term average levels. Similar to the behaviour of a rubber band, stretch too far out and it will snap back.

A key characteristic of the Mean Reversion strategy is that it profits mostly in sideways markets and loses mostly in trending markets

Relative Strength Index (RSI) measures the magnitude of recent price changes and evaluates if a stock has been overbought or oversold. The average gain or loss used in the calculation is the average percentage gain or loss during a look-back period.

'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES'
'''

import pandas as pd
import analyze_stocks_india.supporting_functions as sf #get_moving_avg, load_data
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('display.max_rows', None)

###################################### Parameters ##########################

# for all the nse 100 stocks

#############################################################################
#symbols = sf.nifty_100
#data = sf.load_data(symbols[0])
#print("Checking if Mean Reversion can be applied: ",end='')
#print(sf.check_mean_reversion(symbols[0]))

#symbols = sf.nifty_100
#symbols = sf.good_penny_shares
#symbols = sf.nifty_100 + sf.good_penny_shares

def stocks_to_buy_strategy_mean_reverser(symbols):
    '''
    :param symbols: python list of nse symbols to analyze
    :return: stocks recommend to buy as list
    Shows: details of the stocks and figure showing the plots
    '''
    symbols_buy, rsi_data_list, fig_list, mean_reversion_list = sf.predict_stock_to_buy_using_RSI(symbols,3,30,70, 0.05)


    for i in range(len(symbols_buy)):
        print("Buy Signal for {} using RSI and Mean Reversion is {}".format(symbols_buy[i],mean_reversion_list[i]))


    for fig in fig_list:
        fig.show()

    return symbols_buy


    ###### Just to create plot of all the companies
    #for s in symbols:
    #    fig = sf.create_plot_from_symbol(s,["CLOSE"],s)
    #    fig.show()

    return None




#rsi_data, fig = sf.predict_buy_sell_from_RSI(data,25,75)
#fig.show()

#fig = sf.create_plot(data,["CLOSE","RSI_Signal"],title="SBI Trends")
#fig.show()

#fig = sf.create_plot(data,["priceBymv"],title="SBI Trends")
#fig.show()






