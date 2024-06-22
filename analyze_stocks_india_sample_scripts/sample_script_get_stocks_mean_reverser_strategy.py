# This is a sample script which shows how to filter stocks to buy using mean reverser strategy

from analyze_stocks_india.download_historic_data import download_raw_files, extract_new_data, get_directory
from analyze_stocks_india.Mean_Reverser_Strategy import stocks_to_buy_strategy_mean_reverser
from analyze_stocks_india.nse_symbols import nifty_50, nifty_100, good_penny_shares

symbol = nifty_100

###################################### download and process data first ######################################
# to download raw files till today (specify the start date)
download_raw_files('01/01/2023')
# to process the raw files and extract information (specify the symbol_list and start date)
extract_new_data(symbol, '01/01/2023')

buy_signals = stocks_to_buy_strategy_mean_reverser(symbol)
print('\n\n\n')
print("Buy signal for following stocks")
print(buy_signals)

