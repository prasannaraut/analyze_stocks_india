# This is a sample script which shows how to download raw data files from nse website and extract the processed data from raw files

from analyze_stocks_india.download_historic_data import download_raw_files, extract_new_data, get_directory
from analyze_stocks_india.nse_symbols import nifty_50, nifty_100, nifty_all
import time

print(help(extract_new_data))
print(help(download_raw_files))

# to download raw files till today (specify the start date)
download_raw_files('01/01/2023')

# to process the raw files and extract information (specify the symbol_list and start date)
extract_new_data(nifty_all, '01/01/2023')