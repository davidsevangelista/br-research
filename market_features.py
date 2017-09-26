import pandas as pd
import pickle
import matplotlib.pylab as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')

from fetcher_br_data import *

# Params for plot in LaTeX style
import pylab
from pylab import sqrt

fig_width_pt = 246.0 # get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27                   # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2               # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # width in inches
fig_height = fig_width*golden_mean          # height in inches
fig_size= [fig_width, fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pylab.rcParams.update(params)

fig_path = '/home/evanged/website_HUGO/personalwebsite-hugo/static/img'

# Write pickle data
def write_pickle_data(path_lob, path_cancelled, path_trades, symbol, date):
    # Quotes
    lob_1_one_day = load_lob_zip(path_lob, symbol, date, depth)
    pickle.dump(lob_1_one_day ,open('lob_1_one_day.p', 'wb'))

    # Cancelations
    #cancellations_one_day = load_cancelled_order_zip(path_cancelled, symbol, date)
    #pickle.dump(cancellations_one_day ,open('cancellations_one_day.p', 'wb'))

    # Limit Orders
    #limit_orders_one_day = load_LONEW_zip(path_LO, symbol, date[i])
    #pickle.dump(limit_orders_one_day, open('limit_orders_one_day.p', 'wb'))

    # Trades
    #trades_one_day = load_trades_zip(path_trades, symbol, date)
    #pickle.dump(trades_one_day, open('trades_one_day.p', 'wb'))

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
    imb_num = imb.apply(lambda x: x[0] - x[1], axis=1)
    imb_den = imb.apply(lambda x: x[0] + x[1], axis=1)
    imbalance = pd.Series(imb_num.div(imb_den,axis='index'))#.drop_duplicates().dropna()
    return imbalance


# Plot functions
def plot_imbalance_one_day():
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    get_imbalance(lob).plot()
    return fig

def plot_lob_one_day():
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    lob['Bid Price0'].plot()
    lob['Ask Price0'].plot()
    return fig

def plot_imbalance_reg_grid(Time):
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    to_reg_grid(drop_rep(get_imbalance(lob)),pd.to_timedelta(Time)).plot()
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
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    imbalance = get_imbalance(lob)
    imb_reg_grid = to_reg_grid(drop_rep(imbalance),pd.to_timedelta(Time))
    ax2.plot(imb_reg_grid, 'go-')
    ax2.set_ylabel('Imbalance', color='g')

    ax1.plot(to_reg_grid(drop_rep(lob['Bid Price0']),pd.to_timedelta(Time)), 'b')
    ax1.plot(to_reg_grid(drop_rep(lob['Ask Price0']),pd.to_timedelta(Time)),'r')
    ax1.set_ylabel('Bid Price (blue)/Ask Price (red)', color='k')

    return fig

# Join Lob, Trades and order imbalance
def plot_lob_trades_imbalance(lob, trades):
    # Join lob and trades in the same time
    lob_trade = drop_rep(lob.join(trades[["Price", "Volume", \
                        "Buy Broker", "Sell Broker"]], how='outer'))

    trade_bid_side = lob_trade[lob_trade["Price"] == lob_trade["Bid Price0"]]
    trade_ask_side = lob_trade[lob_trade["Price"] == lob_trade["Ask Price0"]]

    lob_trade = lob_trade.join(trade_bid_side["Price"], how='outer', rsuffix=' Trade'+ ' Bid')
    lob_trade = lob_trade.join(trade_ask_side["Price"], how='outer', rsuffix=' Trade'+ ' Ask')

    imbalance = get_imbalance(lob_trade)
    imbalance = imbalance[(imbalance < -0.95)]# & (imbalance > 0.5) ]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax2 = ax1.twinx()
    ax1.plot(lob_trade["Bid Price0"],linewidth=1)
    ax1.plot(lob_trade["Ask Price0"],linewidth=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Bid"], \
            s=2*lob_trade["Volume"], color='blue',  alpha=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Ask"], \
            s=2*lob_trade["Volume"], color='red', alpha=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Bid"], \
            s=2*lob_trade["Bid Volume0"],color='blue', alpha=0.2)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Ask"], \
            s=2*lob_trade["Ask Volume0"],color='red', alpha=0.2)


    ax1.set_ylabel('Bid Price (blue)/Ask Price (red)/Trades (black)', color='k')

    #ax2.plot(imbalance,'g-', linewidth=1)
    #ax2.set_ylabel('Imbalance', color='g')

    return fig, lob_trade

# Functions for animation

#def main():
#    numframes = 100
#    numpoints = 10
#    color_data = np.random.random((numframes, numpoints))
#    x, y, c = np.random.random((3, numpoints))
#
#    fig = plt.figure()
#    scat1 = plt.scatter(x, y, c=c, s=100)
#    scat2 = plt.scatter(x, y, c=c, s=100)
#
#    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
#                                  fargs=(color_data, scat))
#    plt.show()

def update_plot(i, data, scat4):#, scat2, scat3, scat4):
    #scat1.set_array(np.array(data[i]))
    #scat2.set_array(np.array(data[i]))
    #scat3.set_array(np.array(data[i]))
    scat4.set_array(np.array(data[i]))
    return scat4, # scat2, scat3, scat4


# Animation Lob and Trades proportional to volume
def animation_lob_trades_imbalance(lob, trades, numframes):
    lob, cancellations, trades = load_pickle_data() # Load data
    spread = lob["Ask Price0"] - lob["Bid Price0"]
    lob = lob[(spread > 0) & (spread < 5*spread.median())]

    # Join lob and trades in the same time
    lob_trade = drop_rep(lob.join(trades[["Price", "Buy Broker", "Sell Broker"]], how='outer'))

    trade_bid_side = lob_trade[lob_trade["Price"] == lob_trade["Bid Price0"]]
    trade_ask_side = lob_trade[lob_trade["Price"] == lob_trade["Ask Price0"]]

    lob_trade = lob_trade.join(trade_bid_side["Price"], how='outer', rsuffix=' Trade'+ ' Bid')
    lob_trade = lob_trade.join(trade_ask_side["Price"], how='outer', rsuffix=' Trade'+ ' Ask')

    imbalance = get_imbalance(lob_trade)
    imbalance = imbalance[(imbalance < -0.95)]# & (imbalance > 0.5) ]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax2 = ax1.twinx()
    #scat1 = ax1.plot(lob_trade["Bid Price0"],linewidth=1)
    #scat2 = ax1.plot(lob_trade["Ask Price0"],linewidth=1)
    #scat3 = ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Bid"], \
    #        s=lob_trade["Bid Volume0"]/lob_trade["Bid Volume0"].min())
    scat4 = ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Ask"], \
            s=lob_trade["Ask Volume0"]/lob_trade["Ask Volume0"].min())
    ax1.set_ylabel('Bid Price (blue)/Ask Price (red)/Trades (black)', color='k')

    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(lob_trade, scat4))#, scat2, scat3, scat4))
    plt.show()

    #ax2.plot(imbalance,'g-', linewidth=1)
    #ax2.set_ylabel('Imbalance', color='g')

    #return fig, lob_trade

def plot_lob_trades_imbalance_reg_grid(lob, trades, Time):
    lob, cancellations, trades = load_pickle_data() # Load data
    spread = lob["Ask Price0"] - lob["Bid Price0"]
    lob = lob[(spread > 0) & (spread < 5*spread.median())]

    lob_trade = lob.join(trades[["Price", "Buy Broker", "Sell Broker"]], how='outer')
    imbalance = drop_rep(get_imbalance(lob_trade))
    imbalance = imbalance[(imbalance < -0.95) | (imbalance > 0.95)]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(to_reg_grid(drop_rep(lob_trade["Bid Price0"]), pd.to_timedelta(Time)),linewidth=1)
    ax1.plot(to_reg_grid(drop_rep(lob_trade["Ask Price0"]), pd.to_timedelta(Time)),linewidth=1)
    ax1.plot(to_reg_grid(drop_rep(lob_trade["Price"]), pd.to_timedelta(Time)),'k*', linewidth=0.1)
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
fig_lob_trades_imbalance, lob_trade = plot_lob_trades_imbalance(lob, trades); fig_lob_trades_imbalance.show()
#fig_lob_trades_imbalance_reg_grid = plot_lob_trades_imbalance_reg_grid(lob, trades, '10m'); fig_lob_trades_imbalance_reg_grid.show()

#animation_lob_trades_imbalance(lob, trades, 100)
