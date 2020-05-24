from pymongo import MongoClient
from pymongo import UpdateOne
import csv

client = MongoClient(port = 27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze
col_stock = db.Stock
with open('NVDA.csv','r') as nvda:
    nvda.readline()
    nvda_read = csv.reader(nvda, delimiter = ',')
    for i in nvda_read:
        date = str(i[0])
        open = float(i[1])
        high = float(i[2])
        low = float(i[3])
        close = float(i[4])
        adj_close = float(i[5])
        volume = int(i[6])
        ins = {"date": date, "open source" : open, "high price": high, "low price": low, "close price": close, "adjacent close price": adj_close, "volume": volume}
        col_stock.insert_one(ins)

