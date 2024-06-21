Objective
    Library of functions for stock price trend prediction

Python Requirement
    version >= 3.6

Packages requirement
    As mentioned in requirements.txt

Stock Files
    All the downloaded files are in directory data_files --> raw <br>
    All the processed data files are in directory data_files --> processed <br>
    Note: After installation of library 'data_files' folder will not be present, once you start downloading files these will be available at the same location where this library is installed.<br>

Sample Scripts
    Few sample scripts have been created in sample_scripts folder for ease of understanding, available in the same directoyr where this library is installed.

Library Packages
    create_plot.py --> to create plots of stock trend and trading volume <br />
    download_historic_data.py --> to download data files from nse webseite and process them<br />
    get_stock_moment_from_optionChain.py --> Recomends stocks using option chain data<br />
    Mean_Reverser_Strategy.py --> Apply Mean Reverser Strategy and recommend stocks<br />
    news_sentiment_analysis_Indian_Stocks.py --> performs a sentiment analysis on stock based on google news headline from last 7 days<br />

Extra Information for different types of data from NSE website.
	SYMBOL: Symbol of the listed company.<br />
	SERIES: Series of the equity. Values are [EQ, BE, BL, BT, GC and IL]<br />
	OPEN: The opening market price of the equity symbol on the date.<br />
	HIGH: The highest market price of the equity symbol on the date.<br />
	LOW: The lowest recorded market price of the equity symbol on the date.<br />
	CLOSE: The closing recorded price of the equity symbol on the date.<br />
	LAST: The last traded price of the equity symbol on the date.<br />
	PREVCLOSE: The previous day closing price of the equity symbol on the date.<br />
	TOTTRDQTY: Total traded quantity of the equity symbol on the date.<br />
	TOTTRDVAL: Total traded volume of the equity symbol on the date.<br />
	TIMESTAMP: Date of record.<br />
	TOTALTRADES: Total trades executed on the day.<br />
	ISIN: International Securities Identification Number.<br />
