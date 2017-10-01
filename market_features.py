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

factor=5 # 1 is for latex image in latex file

fig_width_pt = 246.0*factor # get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27                   # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2               # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # width in inches
fig_height = fig_width*golden_mean          # height in inches
fig_size= [fig_width, fig_height]
params = {'backend':         'ps',
          'axes.labelsize':  10*(factor/2),
          'font.size':       10*(factor/2),
          'legend.fontsize': 10*(factor/2),
          'xtick.labelsize':  8*(factor/2),
          'ytick.labelsize':  8*(factor/2),
          'text.usetex':      True,
          'figure.figsize':   fig_size}
pylab.rcParams.update(params)

fig_path = '/home/evanged/website_HUGO/personalwebsite-hugo/static/img/'

# Write pickle data
def write_pickle_data(path_lob, path_cancelled, path_trades, symbol, date):
    # Quotes
    lob_1_one_day = load_lob_zip(path_lob, symbol, date, depth)
    pickle.dump(lob_1_one_day, open('lob_1_one_day.p', 'wb'))

    # Cancelations
    cancellations_one_day = load_cancelled_order_zip(path_cancelled, symbol, date)
    pickle.dump(cancellations_one_day, open('cancellations_one_day.p', 'wb'))

    # Limit Orders
    #limit_orders_one_day = load_LONEW_zip(path_LO, symbol, date[i])
    #pickle.dump(limit_orders_one_day, open('limit_orders_one_day.p', 'wb'))

    # Trades
    trades_one_day = load_trades_zip(path_trades, symbol, date)
    pickle.dump(trades_one_day, open('trades_one_day.p', 'wb'))

# Load pickle data
def load_pickle_data():
    lob_1_one_day = pickle.load(open('lob_1_one_day.p', 'rb'))
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
    imb = quotes[['Bid Volume0', 'Ask Volume0']]\
            .apply(lambda x: (x[0] - x[1])/(x[0] + x[1]), axis=1)
    imbalance = pd.Series(imb)
    return imbalance

# Plot functions
def plot_imbalance_one_day(lob, trades):
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
    to_reg_grid(drop_rep(lob['Bid Price0']), pd.to_timedelta(Time)).plot()
    to_reg_grid(drop_rep(lob['Ask Price0']), pd.to_timedelta(Time)).plot()
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

def plot_spread_midprice(lob):
    spread = lob[["Ask Price0", "Bid Price0"]].apply(lambda x: x[0]-x[1], axis=1)
    spread_tick = spread/spread.min()
    mid_price = lob[["Ask Price0", "Bid Price0"]].mean(axis=1)

    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax2.plot(spread,'r',label='Spread')
    ax2.set_ylabel('Spread')
    ax2.set_ylim([0,10])
    ax1.plot(mid_price, 'b', label='Mid-price')
    ax1.set_ylabel('Mid Price')
    pylab.title('Spread in ticks and mid-price')
    lines = ax1.get_lines() + ax2.get_lines()
    pylab.legend(lines, [l.get_label() for l in lines], loc='upper center')
    pylab.savefig(fig_path + 'plot_sread_midprice.png')
    return fig

# Features: imbalance part 2
def get_lob_trade_imbalance(lob, trades):

    # Join lob and trades at the same time stamp
    lob_trade = drop_rep(lob.join(trades[["Price", "Volume", \
                        "Buy Broker", "Sell Broker"]], how='outer'))

    # Fill lob/quotes if trade occurs in a time stamp is no available for quotes
    columns_to_fill = ['Bid Price0', 'Ask Price0',
                       'Bid Volume0', 'Ask Volume0']
    lob_trade[columns_to_fill] = lob_trade[columns_to_fill].fillna(method='ffill')

    # Separate trades by side
    trade_sell_mo = lob_trade[lob_trade["Price"] == lob_trade["Bid Price0"]]
    trade_buy_mo = lob_trade[lob_trade["Price"] == lob_trade["Ask Price0"]]

    # Join lob with buy and sell market orders (MO)
    lob_trade = lob_trade.join(trade_sell_mo["Price"], how='outer', \
            rsuffix=' Trade Sell MO')

    lob_trade = lob_trade.join(trade_buy_mo["Price"], how='outer', \
            rsuffix=' Trade Buy MO')

    # Join lob with imbalance
    imbalance = pd.DataFrame(get_imbalance(lob))
    imbalance.columns=['Imbalance']

    # Update lob_trade
    lob_trade = lob_trade.join(imbalance, how = 'outer')

    # Make sure there is no imbalance with NaN
    lob_trade['Imbalance'] = lob_trade['Imbalance'].fillna(method='ffill')

    # Join lob with imbalance when trade occurs
    imbalance_trade = lob_trade.dropna(subset=['Price'])
    imbalance_trade = pd.DataFrame(imbalance_trade['Imbalance'])
    imbalance_trade.columns=['Imbalance Trade']

    # Update lob_trade
    lob_trade = lob_trade.join(imbalance_trade, how='outer')

    # Join imbalance corrensponding to the side of each trade
        # Buy side
    volume_when_buy_MO = lob_trade.dropna(subset=\
            ['Price Trade Buy MO'])[['Bid Volume0', 'Ask Volume0']]

    imbalance_when_buy_MO = pd.DataFrame(volume_when_buy_MO\
            .apply(lambda x: (x[0] - x[1])/(x[0] + x[1]), axis=1))
    imbalance_when_buy_MO.columns=['Imbalance Buy MO']

    # Update lob_trade
    lob_trade = lob_trade.join(imbalance_when_buy_MO, how='outer') #Join

        # Sell side
    volume_when_sell_MO = lob_trade.dropna(subset=\
            ['Price Trade Sell MO'])[['Bid Volume0', 'Ask Volume0']]

    imbalance_when_sell_MO = pd.DataFrame(volume_when_sell_MO\
            .apply(lambda x: (x[0] - x[1])/(x[0] + x[1]), axis=1))
    imbalance_when_sell_MO.columns=['Imbalance Sell MO']

    # Update lob_trade
    lob_trade = lob_trade.join(imbalance_when_sell_MO, how='outer') # Join

    # Build imbalance divided by regime
    imbalance = lob_trade['Imbalance']
    regimes = ['Regime 1',
               'Regime 2',
               'Regime 3',
               'Regime 4',
               'Regime 5']

    # Ajust lob and trades to accomodate bins by regime
    #bins = [-1, -0.5, -0.2, 0.2, 0.6, 1]

    # retrive bins automatically: retbins, by regimes
    imbalance_reg = pd.cut(imbalance, len(regimes), retbins=True, labels = regimes)
    imbalance_regime = pd.DataFrame(imbalance_reg[0])
    imbalance_regime.columns=["Regime"]

    # Join lob_trade with Regime
    lob_trade = lob_trade.join(imbalance_regime, how='outer')
    return lob_trade, imbalance_reg[1]

def plot_imbalance_trades(lob, trades):
    lob_trade, imb_bins = get_lob_trade_imbalance(lob, trades)

    start = 50000
    end = 50500
    lob_trade = lob_trade.iloc[start:end]

    plt.plot(lob_trade['Imbalance Buy MO'], 'b*', label = 'Buy MO')
    plt.plot(lob_trade['Imbalance Sell MO'], 'r*', label = 'Sell MO')
    plt.plot(lob_trade['Imbalance'],'k-', alpha=0.4, label = 'Imbalance')
    plt.ylabel('Imbalance')
    plt.legend()
    plt.title('Order imbalance and market orders')

    # Plot horizontal lines with imbalance_reg[1]
    for lin in imb_bins:
        plt.axhline(y=lin, color='r', linestyle='--', alpha=0.2)

    #plt.savefig(fig_path + 'plot_imbalance_trades.png')
    plt.show()

def imbalance_regime(lob, trades, save=False):
    # Get lob, trade, imbalance, and trades separeted by side (sell MO and buy MO)
    lob_imb_reg, imb_bins = get_lob_trade_imbalance(lob, trades)

    regimes = ['Regime 1',
               'Regime 2',
               'Regime 3',
               'Regime 4',
               'Regime 5']

    # Count number and percentage of sell MO by(conditional to) imbalance regime
    n_sell_MO = [lob_imb_reg["Price Trade Sell MO"][\
            lob_imb_reg["Regime"]== reg].count() for reg in regimes]
    perc_sell_MO = n_sell_MO/sum(n_sell_MO)

    # Count number and percentage of buy MO by(conditional to) imbalance regime
    n_buy_MO = [lob_imb_reg["Price Trade Buy MO"][\
            lob_imb_reg["Regime"]== reg].count() for reg in regimes]
    perc_buy_MO = n_buy_MO/sum(n_buy_MO)

    # Total number of trades conditional to imbalance regime
    n_total_MO = n_sell_MO + n_buy_MO
    perc_total_MO = perc_sell_MO + perc_buy_MO

    # Plot results
    fig, ax = plt.subplots()
    index = np.arange(len(regimes))
    bar_width = 1/8
    opacity = 0.5

    buy_regimes = plt.bar(index, perc_buy_MO, bar_width,
                          alpha=opacity,
                          color='b',
                          label='Buy market orders')

    sell_regimes = plt.bar(index + bar_width, perc_sell_MO, bar_width,
                          alpha=opacity,
                          color='r',
                          label='Sell market orders')

    total_regimes = plt.bar(index + 2*bar_width, perc_total_MO, bar_width,
                          alpha=opacity,
                          color='g',
                          label='Total')
    plt.xlabel('Regime')
    plt.ylabel('\% Market Orders')
    plt.title('Percentage of market orders conditioned to imbalance')
    plt.xticks(index + 2*bar_width, ('1','2','3','4','5'))
    plt.legend()
    plt.tight_layout()

    if save==True:
        plt.savefig(fig_path + 'plot_percentage_imbalance_regimes.png')
    #plt.show()

    #def write_table_imbalance_regime(lob,trades):

    #lob_imb_reg = get_lob

    # Arrival rate: counting the time the imbalance spent in each regime

    aggregation = {'Imbalance Buy MO':{'Num':'count','Mean':'mean'},\
                  'Imbalance Sell MO':{'Num':'count','Mean':'mean'}}

    group_regime = lob_imb_reg[['Imbalance Buy MO',\
                                'Imbalance Sell MO',\
                                'Regime']].groupby('Regime').agg(aggregation)


    total_sum = group_regime.iloc[:, group_regime.columns.get_level_values(1)=='Num'].sum(axis=1)
    total_mean = group_regime.iloc[:, group_regime.columns.get_level_values(1)=='Mean'].mean(axis=1)
    total_dict = {'Num':total_sum,'Mean':total_mean}
    total = pd.concat(total_dict.values(), keys=total_dict.keys(), axis=1)

    cols = pd.MultiIndex.from_tuples([('Total Imbalance','Num'),('Total Imbalance','Mean')])
    total = pd.DataFrame(total.values, columns=cols, index=total.index.values)

    full_group_regime = group_regime.join(total)

    # Number of market orders imbalance
    num_regimes = full_group_regime.columns.get_level_values(1)=='Num'
    full_group_regime.iloc[:,num_regimes].plot(kind='bar',color=['b','r','g'],\
        alpha=0.6, title='Arrival MO Conditional to Imbalance');
    plt.legend();


    # Mean/Average imbalance
    mean_regimes = full_group_regime.columns.get_level_values(1)=='Mean'
    full_group_regime.iloc[:,mean_regimes]\
        .plot(kind='bar',color=['b','r','g'],alpha=0.6, title='Average Imbalance');
    plt.legend();
    plt.show()

    return full_group_regime

def pandas_df_to_markdown_table(df):
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt],columns=df.columns)
    df_formatted = pd.concat([df_fmt,df])
    display(Markdown(df_formatted.to_csv(sep="|", index=False)))

# Join Lob, Trades and order imbalance
def plot_lob_trades_volume(lob, trades):

    lob_trade, imb_bins = get_lob_trade_imbalance(lob, trades)

    start = 150000
    end = 200500
    lob_trade = lob_trade.iloc[start:end]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(lob_trade["Bid Price0"],linewidth=1)
    ax1.plot(lob_trade["Ask Price0"],linewidth=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Sell MO"], \
            s=2*lob_trade["Volume"], color='blue',  alpha=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Buy MO"], \
            s=2*lob_trade["Volume"], color='red', alpha=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Sell MO"], \
            s=2*lob_trade["Bid Volume0"],color='blue', alpha=0.2)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Buy MO"], \
            s=2*lob_trade["Ask Volume0"],color='red', alpha=0.2)

    ax1.set_ylabel('Price', color='k')
    plt.title('Trades and Best Limits')
    plt.savefig(fig_path + 'plot_lob_trades_volume_prop.png')
    plt.show()


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
#plot_lob_trades_volume(lob, trades); fig_lob_trades_volume.show()
#fig_lob_trades_imbalance_reg_grid = plot_lob_trades_imbalance_reg_grid(lob, trades, '10m'); fig_lob_trades_imbalance_reg_grid.show()

#res = plot_spread_midprice(lob)
#imbalance_regime(lob, trades)
#lob_imb, imblob, imblob_nona = plot_imbalance_trades(lob, trades)


#plot_lob_trades_volume(lob, trades)

