import pandas as pd
import pickle
import matplotlib.pylab as plt
from matplotlib import style
style.use('fivethirtyeight')

from fetcher_br_data import *

# Write data in pickle format
def write_pickle_data(path_lob, path_cancelled, path_trades, symbol, date):
    # Quotes
    lob_1_one_day = load_lob_zip(path_lob, symbol, date, depth)
    pickle.dump(lob_1_one_day ,open('lob_1_one_day.p', 'wb'))

    # Cancelations
    cancellations_one_day = load_cancelled_order_zip(path_cancelled, symbol, date)
    pickle.dump(cancellations_one_day ,open('cancellations_one_day.p', 'wb'))

    # Limit Orders
    #limit_orders_one_day = load_LONEW_zip(path_LO, symbol, date[i])
    #pickle.dump(limit_orders_one_day, open('limit_orders_one_day.p', 'wb'))

    # Trades
    trades_one_day = load_trades_zip(path_trades, symbol, date)
    pickle.dump(trades_one_day, open('trades_one_day.p', 'wb'))

# Load pickle data
def load_pickle_data():
    lob_1_one_day = pickle.load( open('lob_1_one_day.p', 'rb'))
    cancellations_one_day = pickle.load( open('cancellations_one_day.p', 'rb'))
    #limit_orders_one_day = pickle.load(open('limit_orders_one_day.p', 'rb'))
    trades_one_day = pickle.load(open('trades_one_day.p', 'rb'))
    return (lob_1_one_day, cancellations_one_day, trades_one_day)

# Fundamental functions for time series in high frequency
def drop_rep(series):
    return series.groupby(series.index).last()
def to_resolution(t, dt):
    return np.ceil(t/dt)*dt
def to_reg_grid(s, dt):
    t0 = np.datetime64(0,'D')
    index = pd.Index(to_resolution(s.index.values - t0, dt) + t0, name = s.index.name)
    r = pd.Series(s.index.values,index).groupby(level=0).max()
    return pd.Series(s[r].values, r.index, name=s.name)



# Market feature: imbalance
def get_imbalance(quotes):
    imb = quotes[['Bid Volume0', 'Ask Volume0']]
    imb_num = imb.apply(lambda x: x[1] - x[0], axis=1)
    imb_den = imb.apply(lambda x: x[0] + x[1], axis=1)
    imbalance = pd.Series(imb_num.div(imb_den,axis='index'))#.drop_duplicates().dropna()
    return imbalance


# Plot functions
def plot_imbalance_one_day():
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    get_imbalance(lob).plot()
    return fig

def plot_imbalance_reg_grid(Time):
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    to_reg_grid(drop_rep(get_imbalance(lob)),pd.to_timedelta(Time)).plot()
    return fig

def plot_lob_one_day():
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    lob['Bid Price0'].plot()
    lob['Ask Price0'].plot()
    return fig

def plot_lob_reg_grid(Time):
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    to_reg_grid(drop_rep(lob['Bid Price0']),pd.to_timedelta(Time)).plot()
    to_reg_grid(drop_rep(lob['Ask Price0']),pd.to_timedelta(Time)).plot()
    return fig

def plot_lob_and_imbalance_reg_grid(Time):

    lob, cancellations, trades = load_pickle_data() # Load data
    # Plot Bid and Ask prices with imbalance. Two axes
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twinx()

    imbalance = get_imbalance(lob)
    imb_reg_grid = to_reg_grid(drop_rep(imbalance),pd.to_timedelta(Time))
    ax2.plot(imb_reg_grid, 'go-')
    ax2.set_ylabel('Imbalance', color='g')

    ax1.plot(to_reg_grid(drop_rep(lob['Bid Price0']),pd.to_timedelta(Time)), 'b')
    ax1.plot(to_reg_grid(drop_rep(lob['Ask Price0']),pd.to_timedelta(Time)),'r')
    ax1.set_ylabel('Bid Price (blue)/Ask Price (red)', color='k')

    return fig1

# Join Lob and Trades
def plot_lob_trades_imbalance(lob, trades):
    lob, cancellations, trades = load_pickle_data() # Load data
    spread = lob["Ask Price0"] - lob["Bid Price0"]
    lob = lob[spread > 0 & ( spread < 5*spread.median())]
    lob_trade = lob.join(trades[["Price", "Buy Broker", "Sell Broker"]], how='outer')
    imbalance = drop_rep(get_imbalance(lob_trade))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(drop_rep(lob_trade["Bid Price0"]),linewidth=1)
    ax1.plot(drop_rep(lob_trade["Ask Price0"]),linewidth=1)
    ax1.plot(drop_rep(lob_trade["Price"]),'k*', linewidth=0.1)
    ax1.set_ylabel('Bid Price (blue)/Ask Price (red)/Trades (black)', color='k')

    ax2.plot(imbalance,'g-', linewidth=1)
    ax2.set_ylabel('Imbalance', color='g')

    return fig

###################################TESTING AREA####################################################
path_lob ='/home/evanged/Dropbox/Work-Research/Finance/LOB Study Data/QuotesByLevel/'
path_trades = '/home/evanged/Dropbox/Work-Research/Finance/NewMarketData/Trades/'
path_cancelled = '/home/evanged/Dropbox/Work-Research/Finance/NewMarketData/CanceledOrders/'
symbol = "DOLJ17"
starttime = '10:00:00.000'
endtime = '17:00:00.000'
depth = 1
date = [datetime.datetime(2017, 3, 10)]
i = 0

#write_pickle_data(path_lob, path_cancelled, path_trades, symbol, date[i])
lob, cancellations, trades = load_pickle_data()
#plot_imbalance_one_day()

#fig_imb = plot_imbalance_reg_grid('10m'); fig_imb.show()
#fig_lob = plot_lob_one_day(); fig_lob.show()
#fig_lob_reg_grid = plot_lob_reg_grid('10m'); fig_lob_reg_grid.show()
#fig_lob_imb = plot_lob_and_imbalance_reg_grid('5m'); fig_lob_imb.show()
fig_lob_trades = plot_lob_trades_imbalance(lob, trades); fig_lob_trades.show()
#lob_trades_spread = plot_lob_and_trades_imbalance(lob, trades);
