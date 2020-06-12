import bs4
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import threading as thd
import time
from pymongo import MongoClient
from pymongo import UpdateOne
import datetime
import threading
import sys

def crawl_price(url):
    try:
        page = urlopen(url)
        soup = bs4.BeautifulSoup(page, 'html.parser')
    except:
        print('Error opening the URL')
        pass



    result = soup.find('div', {'class': 'My(6px) Pos(r) smartphone_Mt(6px)'}).find('span').text

    return result

def relax(start_time):
    while True:
        now_time = datetime.datetime.now()
        if now_time >= start_time:
            print('start crawl')
            break
        print('not now')
        time.sleep(1)

def start_run(end_time, url,collection):
    while True:
        try:

            result = crawl_price(url)

            now_time = datetime.datetime.now()

            ins = {"time":now_time, 'price': result}
            collection.insert_one(ins)
            print(now_time, result)
            if now_time >= end_time:
                print('end crawl')
                break
            time.sleep(5)
        except:
            pass

client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze
nvda = db['nvda_daily']
start_time = datetime.datetime(2020,6,11,9,30,00)
end_time = datetime.datetime(2020,6,11,16,0,0)

def begin_crawl(url, name):
    collection = db[name]
    relax(start_time)
    start_run(end_time, url, collection)

# begin_crawl('https://finance.yahoo.com/quote/NVDA?p=NVDA&.tsrc=fin-srch')

if __name__ == '__main__':
    data = {'AAPL':'https://finance.yahoo.com/quote/AAPL?p=AAPL&.tsrc=fin-srch',
            'NVDA':'https://finance.yahoo.com/quote/NVDA?p=NVDA&.tsrc=fin-srch',
            'TSLA':'https://finance.yahoo.com/quote/TSLA?p=TSLA&.tsrc=fin-srch',
            'FB':'https://finance.yahoo.com/quote/FB?p=FB&.tsrc=fin-srch',
            'MSFT':'https://finance.yahoo.com/quote/MSFT?p=MSFT&.tsrc=fin-srch'}
    input = sys.argv[1]
    web = data[input]
    print(input, web)
    begin_crawl(web, input)

