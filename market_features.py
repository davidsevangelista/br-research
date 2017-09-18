import pandas as pd
import pickle
import matplotlib.pylab as plt
from matplotlib import style
style.use('fivethirtyeight')

from fetcher_br_data import *

path_lob ='/home/evanged/Dropbox/Work-Research/Finance/LOB Study Data/QuotesByLevel/'
path_trades = '/home/evanged/Dropbox/Work-Research/Finance/NewMarketData/Trades/'
path_cancelled = '/home/evanged/Dropbox/Work-Research/Finance/NewMarketData/CanceledOrders/'
symbol = "DOLJ17"
starttime = '10:00:00.000'
endtime = '17:00:00.000'
depth = 1
date = [datetime.datetime(2017, 3, 10)]
i = 0

#lob_1 = load_lob_zip(path_lob, symbol, date[i], depth)
#pickle.dump(lob_1 ,open('lob_1.p', 'wb'))
lob_1 = pickle.load( open('lob_1.p', 'rb'))

#cancellations = load_cancelled_order_zip(path_cancelled, symbol, date[i])
#pickle.dump(cancellations ,open('cancellations.p', 'wb'))
cancellations = pickle.load( open('cancellations.p', 'rb'))

#limit_orders = load_LONEW_zip(path_LO, symbol, date[i])
#pickle.dump(lo_evt, open('limit_orders.p', 'wb'))
#limit_orders = pickle.load(open('limit_orders.p', 'rb'))

#trades = load_trades_zip(path_trades, symbol, date[i])
#pickle.dump(trades, open('trades.p', 'wb'))
trades = pickle.load(open('trades.p', 'rb'))

plt.figure()
plt.plot(lob_1['Bid Price']['price0'])
plt.plot(lob_1['Ask Price']['price0'])
#plt.plot(trades['Price'])
plt.show()

plt.figure()
plt.plot(trades['Price'])
plt.show()


