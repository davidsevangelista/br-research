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

factor=7 # 2 is for latex image in latex file

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
    return series.groupby(series.index).first()
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
    plt.show()

def plot_lob_one_day():
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    lob['Bid Price0'].plot()
    lob['Ask Price0'].plot()
    plt.show()

def plot_imbalance_reg_grid(Time):
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    to_reg_grid(drop_rep(get_imbalance(lob)),pd.to_timedelta(Time)).plot()
    plt.show()

def plot_lob_reg_grid(Time):
    lob, cancellations, trades = load_pickle_data() # Load data
    fig = plt.figure()
    to_reg_grid(drop_rep(lob['Bid Price0']), pd.to_timedelta(Time)).plot()
    to_reg_grid(drop_rep(lob['Ask Price0']), pd.to_timedelta(Time)).plot()
    plt.show()

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
    plt.show()


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
    plt.show()

# Features: imbalance part 2
def get_lob_trade_imbalance(lob, trades):

    lob = lob.shift(-1)
    # Join lob and trades at the same time stamp
    trades = trades[["Price", "Volume", "Buy Broker", "Sell Broker",
                     "Buy Agressor", "Sell Agressor", "Cross Indicator"]]#.dropna(subset=['Price'])

    # This is to exclude auctions and agressive buy and sell
    # at the same Report Time
    trades = trades[~(((trades["Cross Indicator"]==1)
                      |(trades["Cross Indicator"]==0))
                      &(trades["Buy Agressor"]==2)
                      &(trades["Sell Agressor"]==2))]

    # Separate trades by side

    trade_sell_mo = trades[(trades["Sell Agressor"]==1) & (trades["Buy Agressor"]==2)]
    trade_buy_mo  = trades[(trades["Sell Agressor"]==2) & (trades["Buy Agressor"]==1)]

    # Join lob with buy and sell market orders (MO)
    trades = trades.join(trade_sell_mo["Price"], how='outer', \
            rsuffix=' Trade Sell MO')

    trades = trades.join(trade_buy_mo["Price"], how='outer', \
            rsuffix=' Trade Buy MO')
    lob_trade = trades.join(lob,how='outer')#.fillna(method='ffill')

    # Join lob with imbalance
    imbalance = pd.DataFrame(get_imbalance(lob_trade))
    imbalance.columns=['Imbalance']

    # Update lob_trade
    imbalance2 = imbalance.reset_index()
    lob_trade2 = lob_trade.reset_index()
    #lob_trade = lob_trade2.merge(imbalance2.\
    #        drop_duplicates(subset=['Report Time']), how='left',on=['Report Time'])
    lob_trade = imbalance2.merge(lob_trade2.\
            drop_duplicates(subset=['Report Time']), how='left',on=['Report Time'])


    # Make sure there is no imbalance with NaN
    #lob_trade['Imbalance'] = lob_trade['Imbalance'].fillna(method='ffill')

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


    # Maybe I need to improve this part
    # Build imbalance divided by regime
    imbalance = lob_trade['Imbalance']
    regimes = ['Regime 1',
               'Regime 2',
               'Regime 3',
               'Regime 4',
               'Regime 5']

    # Ajust lob and trades to accomodate bins by regime. This is when bins are
    # not automated retrieved from pd.cut
    bins = np.array([-1, -0.6, -0.2, 0.2, 0.6, 1])

    # retrive bins automatically: retbins, by regimes
    #imbalance_reg = pd.cut(imbalance, len(regimes), retbins=True, labels = regimes) # auto
    imbalance_reg = pd.cut(imbalance, bins, labels = regimes)
    #imbalance_regime = pd.DataFrame(imbalance_reg[0]) # auto
    imbalance_regime = pd.DataFrame(imbalance_reg) # auto
    imbalance_regime.columns=["Regime"]

    # Join lob_trade with Regime
    lob_trade = lob_trade.join(imbalance_regime, how='outer')
    return lob_trade, bins#, imbalance_regime#, imbalance_reg[1] auto

def plot_imbalance_trades(lob, trades):
    import datetime
    lob_trade, imb_bins = get_lob_trade_imbalance(lob, trades)
    lob_trade.set_index('Report Time', inplace=True)

    start = 49200
    end = 49900
    lob_trade = lob_trade.iloc[start:end]

    fig, ax = plt.subplots()

    lob_trade['Imbalance'].plot(style='k', alpha=0.4, label = 'Imbalance')

    ax.scatter(lob_trade.index.values, lob_trade['Imbalance Buy MO'],\
            s=15*lob_trade['Volume'], color='blue', label = 'Buy MO')
    ax.scatter(lob_trade.index.values, lob_trade['Imbalance Sell MO'],\
            s=15*lob_trade['Volume'], color='red', label = 'Sell MO')

    plt.ylabel('Imbalance')
    plt.legend()
    plt.title('Order imbalance and market orders')

    # Plot horizontal lines with imbalance_reg[1]
    for lin in imb_bins:
        ax.axhline(y=lin, color='r', linestyle='--', alpha=0.2)

    #fig.autofmt_xdate()
    #ax.set_xlim(xmin=lob_trade.index.values[0],\
    #            xmax=lob_trade.index.values[-1])

    plt.savefig(fig_path + 'plot_imbalance_trades.png')
    plt.show()

def imbalance_regime(lob, trades):
    # Get lob, trade, imbalance, and trades separeted by side (sell MO and buy MO)
    lob_imb_reg, imb_bins = get_lob_trade_imbalance(lob, trades)

    # I still have to update this part, to avoid the future warning
    aggregation = {'Imbalance Buy MO':{'Num':'count','Mean':'mean'},\
                  'Imbalance Sell MO':{'Num':'count','Mean':'mean'}}#,\
                #  'Count':{'Count Orders':'sum'}}

    group_regime = lob_imb_reg[['Imbalance Buy MO',\
                                'Imbalance Sell MO',\
                               # 'Count',\
                                'Regime']].groupby('Regime').agg(aggregation)


    total_sum      = group_regime.iloc[:, group_regime.columns.\
                     get_level_values(1)=='Num'].sum(axis=1)

    total_mean     = group_regime.iloc[:, group_regime.columns.\
                     get_level_values(1)=='Mean'].mean(axis=1)

    #total_num_walk = group_regime.iloc[:,group_regime.columns.\
    #                 get_level_values(1)=='Count Orders'].sum(axis=1)

    total_dict = {'Mean':total_mean, 'Num':total_sum}#, 'Count Orders':total_num_walk}
    total = pd.concat(total_dict.values(), keys=total_dict.keys(), axis=1)

    cols = pd.MultiIndex.from_tuples([('Total Imbalance','Mean'), \
                                      ('Total Imbalance','Num')])#,\
                                     # ('Total Imbalance','Count Orders')])

    total = pd.DataFrame(total.values, columns=cols, index=total.index.values)

    full_group_regime = group_regime.join(total)

    # Number of market orders imbalance
    num_regimes = full_group_regime.columns.get_level_values(1)=='Num'
    num_table = full_group_regime.iloc[:,num_regimes]
    num_table.columns = num_table.columns.droplevel(1)

    num_table.plot(kind='bar',color=['b','r','g'],\
        alpha=0.6, title='Arrival MO Conditional on Imbalance');
    plt.ylabel('Quantity of Market Orders')
    plt.legend();
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(fig_path + 'plot_imbalance_regime_num_MO.png')


    # Percentage of market orders imbalance
    fg_regime_perc=full_group_regime.iloc[:,full_group_regime\
            .columns.get_level_values(1)=='Num'].transform(lambda x:100*x/x.sum())
    fg_regime_perc.columns=fg_regime_perc.columns.droplevel(1)
    perc_table = fg_regime_perc

    perc_table.plot(kind='bar', color=['b','r','g'], alpha=0.6,\
            title='Percentage of Market Orders Conditional on Imbalance')
    plt.ylabel('\% of Market Orders')
    plt.legend()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(fig_path + 'plot_imbalance_regime_perc_MO.png')


    # Mean/Average imbalance

    num_regimes = full_group_regime.columns.get_level_values(1)=='Mean'
    mean_table = full_group_regime.iloc[:,num_regimes]
    mean_table.columns = mean_table.columns.droplevel(1)

    mean_table.plot(kind='bar',color=['b','r','g'],alpha=0.6, title='Average Imbalance');
    plt.legend();
    plt.ylabel('Imbalance')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(fig_path + 'plot_imbalance_regime_mean_MO.png')

    plt.show()

    return full_group_regime

def market_statistics_describe(lob,trades):
    lob_trade, reg = get_lob_trade_imbalance(lob, trades)

    price_lob = lob_trade[['Bid Price0','Ask Price0', 'Bid Volume0', 'Ask Volume0']].describe()
    price_lob.reset_index(level=0, inplace=True)

    price_trades = lob_trade[['Price Trade Buy MO','Price Trade Sell MO', 'Price']].describe()
    price_trades.reset_index(level=0, inplace=True)

    return price_lob, price_trades

def df_to_markdown(df,name):
    import pytablewriter

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = name
    writer.header_list = list(df.columns.values)
    writer.value_matrix = df.values.tolist()
    writer.write_table()

def write_table_imbalance(full_group_regime):

    num_regimes = full_group_regime.columns.get_level_values(1)=='Num'
    num_table = full_group_regime.iloc[:,num_regimes]
    num_table.columns = num_table.columns.droplevel(1)
    num_table.reset_index(level=0,inplace=True)

    num_regimes = full_group_regime.columns.get_level_values(1)=='Mean'
    mean_table = full_group_regime.iloc[:,num_regimes]
    mean_table.columns = mean_table.columns.droplevel(1)
    mean_table.reset_index(level=0,inplace=True)

    fg_regime_perc=full_group_regime.iloc[:,full_group_regime\
            .columns.get_level_values(1)=='Num'].transform(lambda x:100*x/x.sum())
    fg_regime_perc.columns=fg_regime_perc.columns.droplevel(1)
    perc_table = fg_regime_perc
    perc_table.reset_index(level=0,inplace=True)

    return num_table, mean_table, perc_table

# Join Lob, Trades and order imbalance
def plot_lob_trades_volume(lob, trades):

    lob_trade, imb_bins = get_lob_trade_imbalance(lob, trades)
    lob_trade.set_index('Report Time', inplace=True)

    start = 48200
    end = 52500
    lob_trade = lob_trade.iloc[start:end]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax2 = ax1.twinx()
    ax1.plot(lob_trade["Bid Price0"], linewidth=1)
    ax1.plot(lob_trade["Ask Price0"], linewidth=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Sell MO"], \
            s=2*lob_trade["Volume"], color='red',  alpha=1)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Buy MO"], \
            s=2*lob_trade["Volume"], color='blue', alpha=1)

    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Sell MO"], \
            s=2*lob_trade["Bid Volume0"],color='red', alpha=0.2)
    ax1.scatter(lob_trade.index.values,  lob_trade["Price Trade Buy MO"], \
            s=2*lob_trade["Ask Volume0"],color='blue', alpha=0.2)

    ax1.set_ylabel('Price', color='k')
    plt.title('Trades and Best Limits')
    plt.legend()
    #plt.savefig(fig_path + 'plot_lob_trades_volume_prop.png')
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
    plt.show()

###################################TESTING AREA####################################################
path_lob ='/home/evanged/Dropbox/Work-Research/Finance/LOB Study Data/QuotesByLevel/'
path_trades = '/home/evanged/Dropbox/Work-Research/Finance/NewMarketData/Trades/'
path_trades_mac = '/Users/evanged/Dropbox/Work-Research/Finance/NewMarketData/Trades/'
path_cancelled = '/home/evanged/Dropbox/Work-Research/Finance/NewMarketData/CanceledOrders/'
symbol = "DOLJ17"
starttime = '10:00:00.000'
endtime = '17:00:00.000'
depth = 1
date = [datetime.datetime(2017, 3, 10)]
i = 0
buy_broker = 'All'

#write_pickle_data(path_lob, path_cancelled, path_trades_mac, symbol, date[i])
lob, cancellations, trades = load_pickle_data()
#plot_imbalance_one_day()
#plot_imbalance_reg_grid('10m')
#plot_lob_one_day()
#plot_lob_reg_grid('10m')
#plot_lob_and_imbalance_reg_grid('5m')
#plot_lob_trades_volume(lob, trades)
#plot_lob_trades_imbalance_reg_grid(lob, trades, '10m')
#res = plot_spread_midprice(lob)
#imbalance_regime(lob, trades)
#plot_lob_trades_volume(lob, trades)

# Generate plots
#get_lob_trade_imbalance(lob,trades)
#regimes= imbalance_regime(lob, trades)
plot_lob_trades_volume(lob,trades)

# Generate tables
#price_lob, price_trades = market_statistics_describe(lob,trades)
#df_to_markdown(price_trades, 'Trading Statistics')
#df_to_markdown(price_lob, 'Limit Order Book Statistics')
#
#
#lob_trade = get_lob_trade_imbalance(lob,trades)
#regimes = imbalance_regime(lob,trades)
#num, mean, perc = write_table_imbalance(regimes)
#
#
#df_to_markdown(num, 'Number of Market Orders')
#df_to_markdown(mean, 'Average Imbalance')
#df_to_markdown(perc, 'Percentage of Makret Orders')

