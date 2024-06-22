import analyze_stocks_india.supporting_functions as rf
import time
import pandas as pd
pd.set_option('display.max_columns', 500)

#sbin = rf.get_stock_info_json("SBIN")
#bajfinance = rf.get_stock_info_json("BAJFINANCE")
#cipla = rf.get_stock_info_json("CIPLA")
#nifty = rf.get_index_info_json("NIFTY")

'''
print(rf.get_3_nearest_strickPrices(sbin))
print(rf.get_nearest3_avgOpenInterest(sbin,"call_option"))
print(rf.get_nearest3_avgOpenInterest(sbin,"put_option"))
print(rf.get_immediate_moment(sbin))
'''
#print(rf.get_immediate_moment(sbin))
#print(rf.get_immediate_moment(bajfinance))

#data = rf.get_OpenIntest_IV_for_allStrikePrice(sbin,1)
#print(rf.get_support(sbin,1))
#print(rf.get_resistance(sbin,1))

#print(rf.get_support(bajfinance,1))
#print(rf.get_resistance(bajfinance,1))

#symbol = rf.nifty_50
#symbol = rf.good_penny_shares

#symbol.remove("CIPLA")
#symbol.remove("COALINDIA")

#symbol =["HCLTECH", "KOTAKBANK", "HDFCLIFE"]

def get_stock_using_optionChain(symbols):
    '''
    :param symbols: python list of symbols
    :return: list of stocks recommended to buy as list
    :shows: figures with charts for identified stocks
    '''

    buy_signals = []
    for s in symbols:
        print("Checking for {}".format(s))
        stock = rf.get_stock_info_json(s)
        if bool(stock):
            moment = rf.get_immediate_moment(stock)
            print("Immediate moment of {} is {}".format(s, moment))

            expiry_moment = rf.get_expiryDate_moment(stock,1) #output: [movement, movement_percent, avg_LTP_pChange, supports, resistances, current_value]
            print("Expiry Date Moment is {} with chace of {}%, avg_LTP_percentChange = {},   supports are {} resistances are {} with current value of {}".format(
                expiry_moment[0], expiry_moment[1], expiry_moment[2], expiry_moment[3], expiry_moment[4], expiry_moment[5]))

            # create plot if moment is Bullish from Open Interest Analysis
            if (expiry_moment[0] == "Bullish"):
                fig = rf.create_plot_from_symbol(s,["CLOSE"],s+" - Bullish in Near Future")
                fig.show()
                buy_signals.append(s)


        else:
            print("Could not download option chain data for {}".format(s))

        print("\n\n")
        time.sleep(10)

    return buy_signals

'''
sbin['records']['data'] #list one entry for each strike priec and expiration date

# for each entry get strike price expiry date and CE PE options data
sbin['records']['data'][0]['strikePrice']
sbin['records']['data'][0]['expiryDate']
sbin['records']['data'][0]['CE']
sbin['records']['data'][0]['PE']


#get data individually for each option
sbin['records']['data'][0]['CE']['openInterest']
sbin['records']['data'][0]['CE']['changeinOpenInterest']
sbin['records']['data'][0]['CE']['impliedVolatility']
sbin['records']['data'][0]['CE']['pChange']
'''