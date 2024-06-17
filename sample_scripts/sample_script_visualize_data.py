# This is a sample script to visualize day closing price of stock as plot
from analyze_stocks_india.create_plots import get_trend_plot, get_trade_quantity

print(help(get_trend_plot))
print(help(get_trade_quantity))

fig = get_trend_plot("SBICARD")
fig.show()

fig = get_trade_quantity("SBICARD")
fig.show()