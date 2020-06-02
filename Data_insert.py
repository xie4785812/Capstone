from pymongo import MongoClient
from pymongo import UpdateOne
from dateutil import parser
import csv

client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze

def add_stock_data(filename):
    name = filename[0:-4]
    print(name)
    col_stock = db[name]
    with open(filename, 'r') as nvda:
        nvda.readline()
        nvda_read = csv.reader(nvda, delimiter=',')
        for i in nvda_read:
            date = parser.parse(i[0])
            open_price = float(i[1])
            high = float(i[2])
            low = float(i[3])
            close = float(i[4])
            adj_close = float(i[5])
            volume = int(i[6])
            ins = {"date": date, "open price": open_price, "high price": high, "low price": low, "close price": close,
                   "adjacent close price": adj_close, "volume": volume}
            col_stock.insert_one(ins)

# add_stock_data('NVDA.csv')


def add_gold_data(filename):
    col_stock = db.Gold
    with open(filename, 'r') as gold:
        gold.readline()
        gold_read = csv.reader(gold, delimiter=',')
        for i in gold_read:
            date = parser.parse(i[0])
            open_price = float(i[1])
            high = float(i[2])
            low = float(i[3])
            close = float(i[4])
            adj_close = float(i[5])
            volume = int(i[6])
            ins = {"date": date, "open price": open_price, "high price": high, "low price": low, "close price": close,
                   "adjacent close price": adj_close, "volume": volume}
            col_stock.insert_one(ins)

add_gold_data('GOLD.csv')

def add_exchange_data(filename):
    name = filename[0:-4]
    col_stock = db[name]
    with open(filename, 'r') as ex:
        ex.readline()
        ex_read = csv.reader(ex, delimiter=',')
        for i in ex_read:
            date = parser.parse(i[0])
            open_price = i[1]
            high = i[2]
            low = i[3]
            close = i[4]
            adj_close = i[5]
            volume = i[6]
            ins = {"date": date, "open price": open_price, "high price": high, "low price": low, "close price": close,
                   "adjacent close price": adj_close, "volume": volume}
            col_stock.insert_one(ins)

add_exchange_data('CNY=X.csv')

