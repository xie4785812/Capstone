import bs4
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import threading as thd
import time

def crawl_price():
    url = 'https://finance.yahoo.com/quote/NVDA?p=NVDA&.tsrc=fin-srch'

    try:
        page = urlopen(url)
    except:
        print('Error opening the URL')

    soup = bs4.BeautifulSoup(page, 'html.parser')

    result = soup.find('div', {'class': 'My(6px) Pos(r) smartphone_Mt(6px)'}).find('span').text

    return result


while True:
    print(crawl_price())
    time.sleep(5)