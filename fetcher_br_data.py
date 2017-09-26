import datetime
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt
import math

cancelled_and_LO_columns = ["Time Frame",
                            "Report Time",
                            "Side",
                            "Order Quantity",
                            "Order Price",
                            "Execution Quantity",
                            "Seq Order Number",
                            "Broker"]
trades_columns = ['Time Frame',
                  'Report Time',
                  'Price',
                  'Volume',
                  'Buy Seq Ord Num',
                  'Buy Agressor',
                  'Sell Seq Ord Num',
                  'Sell Agressor',
                  'Cross Indicator',
                  'Buy Broker',
                  'Sell Broker']

#trades_columns = ["Time Frame",
#		  "Report Time",
#		  "Price",
#		  "Volume",
#		  "Buy Seq Order Number",
#		  "Buy Agressor",
#		  "Sell Seq Order Number",
#		  "Sell Agressor",
#		  "Cross Indicator",
#		  "Buy Broker",
#		  "Sell Broker"]

def loadstr(filename):
    dat = np.loadtxt(filename, dtype=np.str_, delimiter=';')
    for i in range(0,np.size(dat[:,0])):
        for j in range(0,np.size(dat[0,:])):
            mystring = dat[i,j]
            tick = len(mystring) - 1
            dat[i,j] = mystring[2:tick]

    return (dat)

def load_LONEW_zip(path,symbol,date):
    zip_name = path + symbol + '_' + date.strftime("%Y%m%d") + ".zip"
    x = []
    if (Path(zip_name).is_file()):
        zip = zipfile.ZipFile(zip_name)
        file = symbol + "_LONEW_" + date.strftime("%Y%m%d") + ".txt"
        if (file in [i.filename for i in zip.filelist]):
            #x = pd.DataFrame(loadstr(zip.open(file, 'r')))
            x = pd.DataFrame(loadstr(zip.open(file, 'r')),sep=';')
            x = pd.read_csv(loadstr(zip.open(file, 'r')),sep=';')
            x.columns = cancelled_and_LO_columns
            x["Order Quantity"] = pd.to_numeric(x["Order Quantity"],errors='coerce')
            x["Execution Quantity"] = pd.to_numeric(x["Execution Quantity"],errors='coerce')
            x["Order Price"] = pd.to_numeric(x["Order Price"],errors='coerce')
            x["Report Time"] = pd.to_datetime(x["Report Time"])
            x.set_index("Report Time", inplace=True)

    return x

def load_cancelled_order_zip(path,symbol,date):
    zip_name = path + symbol + '_' + date.strftime("%Y%m%d") + ".zip"
    x = []
    if (Path(zip_name).is_file()):
        zip = zipfile.ZipFile(zip_name)
        file = symbol + "_CA_" + date.strftime("%Y%m%d") + ".txt"
        if (file in [i.filename for i in zip.filelist]):
            #x = pd.DataFrame(loadstr(zip.open(file, 'r')))
            x = pd.read_csv(loadstr(zip.open(file, 'r')), sep=';')
            x.columns = cancelled_and_LO_columns
            x["Order Quantity"] = pd.to_numeric(x["Order Quantity"],errors='coerce')
            x["Execution Quantity"] = pd.to_numeric(x["Execution Quantity"],errors='coerce')
            x["Order Price"] = pd.to_numeric(x["Order Price"],errors='coerce')
            x["Report Time"] = pd.to_datetime(x["Report Time"])
            x.set_index("Report Time", inplace=True)

    return x

def load_trades_zip(path,symbol,date):
    zip_name = path + symbol + '_' + date.strftime("%Y%m%d") + ".zip"
    x = []
    if (Path(zip_name).is_file()):
        zip = zipfile.ZipFile(zip_name)
        file = symbol + "_NEG_" + date.strftime("%Y%m%d") + ".txt"
        if (file in [i.filename for i in zip.filelist]):
            #x = pd.DataFrame(loadstr(zip.open(file, 'r')))
            x = pd.read_csv(zip.open(file, 'r'),sep=';')
            x.columns = trades_columns
            x["Volume"] = pd.to_numeric(x["Volume"],errors='coerce')
            x["Price"] = pd.to_numeric(x["Price"],errors='coerce')
            x["Report Time"] = pd.to_datetime(x["Report Time"])
            x.set_index("Report Time", inplace=True)
    return x

def load_lob_zip(path,symbol,date,depth):
    zip_name = path + symbol + '_' + date.strftime("%Y%m%d") + ".zip"

    offer_price = []
    bid_price = []
    offer_qty = []
    bid_qty = []

    if(Path(zip_name).is_file()):
        zip = zipfile.ZipFile(zip_name)
        #Load Sell Price LOB
        file = symbol + "_SellPriceLevel_" + date.strftime("%Y%m%d") + ".txt"
        if(file in [i.filename for i in zip.filelist]):
            f = zip.open(file,'r')
            data = [x.decode().strip('\r\n').split(sep=';') for x in f.readlines()]
            offer_price = pd.DataFrame([data[i][0:min(depth+2, len(data[i]))] if len(data[i]) >= 3 else [] for i in range(0, len(data))])
            offer_price.columns = ['Time Frame', "Report Time"] + ['Ask Price' + str(i) for i in range(0,depth)]
            offer_price = offer_price[offer_price['Ask Price0'] != '']
            for i in range(0,depth):
                offer_price['Ask Price' + str(i)] = pd.to_numeric(offer_price['Ask Price' + str(i)], errors='coerce')
            f.close()
        else:
            offer_price = []

        # Load Buy Price LOB
        file = symbol + "_BuyPriceLevel_" + date.strftime("%Y%m%d") + ".txt"
        if (file in [i.filename for i in zip.filelist]):
            f = zip.open(file,'r')
            data = [x.decode().strip('\r\n').split(sep=';') for x in f.readlines()]
            bid_price = pd.DataFrame(
                [data[i][0:min(depth + 2, len(data[i]))] if len(data[i]) >= 3 else [] for i in
                 range(0, len(data))])
            bid_price.columns = ['Time Frame', "Report Time"] + ['Bid Price' + str(i) for i in range(0, depth)]
            bid_price = bid_price[bid_price['Bid Price0'] != '']
            for i in range(0, depth):
                bid_price['Bid Price' + str(i)] = pd.to_numeric(bid_price['Bid Price' + str(i)], errors='coerce')
            f.close()
        else:
            bid_price = []

        # Load Sell Volume LOB
        file = symbol + "_SellVolumeLevel_" + date.strftime("%Y%m%d") + ".txt"
        if (file in [i.filename for i in zip.filelist]):
            f = zip.open(file,'r')
            data = [x.decode().strip('\r\n').split(sep=';') for x in f.readlines()]
            offer_qty = pd.DataFrame(
                [data[i][0:min(depth + 2, len(data[i]))] if len(data[i]) >= 3 else [] for i in
                 range(0, len(data))])
            offer_qty.columns = ['Time Frame', "Report Time"] + ['Ask Volume' + str(i) for i in range(0, depth)]
            offer_qty = offer_qty[offer_qty['Ask Volume0'] != '']
            for i in range(0, depth):
                offer_qty['Ask Volume' + str(i)] = pd.to_numeric(offer_qty['Ask Volume' + str(i)], errors='coerce')
            f.close()
        else:
            offer_qty = []

        # Load Buy Volume LOB
        file = symbol + "_BuyVolumeLevel_" + date.strftime("%Y%m%d") + ".txt"
        if (file in [i.filename for i in zip.filelist]):
            f = zip.open(file,'r')
            data = [x.decode().strip('\r\n').split(sep=';') for x in f.readlines()]
            bid_qty = pd.DataFrame(
                [data[i][0:min(depth + 2, len(data[i]))] if len(data[i]) >= 3 else [] for i in
                 range(0, len(data))])
            bid_qty.columns = ['Time Frame', "Report Time"] + ['Bid Volume' + str(i) for i in range(0, depth)]
            bid_qty = bid_qty[bid_qty['Bid Volume0'] != '']
            for i in range(0, depth):
                bid_qty['Bid Volume' + str(i)] = pd.to_numeric(bid_qty['Bid Volume' + str(i)], errors='coerce')
            f.close()
        else:
            bid_qty = []

    lob_dict= {'Bid Price': bid_price, 'Ask Price': offer_price, \
               'Bid Volume':bid_qty, 'Ask Volume':offer_qty}
    mlob_price = pd.merge(lob_dict['Bid Price'], lob_dict['Ask Price'], \
            on=["Report Time", 'Time Frame'],how='outer').fillna(method='ffill')
    mlob_volume = pd.merge(lob_dict['Bid Volume'], lob_dict['Ask Volume'], \
            on=["Report Time", 'Time Frame'],how='outer').fillna(method='ffill')
    mlob = pd.merge(mlob_price, mlob_volume,how='inner').fillna(method='ffill').dropna()

    mlob["Report Time"] = pd.to_datetime(mlob["Report Time"])
    mlob.set_index("Report Time", inplace=True)

    spread = mlob["Ask Price0"] - mlob["Bid Price0"]
    mlob = mlob[(spread > 0) & (spread < 5*spread.median())]


    return mlob
