from pymongo import MongoClient
import numpy as np
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm
import datetime

client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze

training_size = 26000
stock_list = ['AAPL','MSFT','FB','TSLA','NVDA']

def collect_data(name):
    msft = db[name]
    msft_cursor = msft.find({})

    msft_data = []
    time = datetime.datetime(2020,6,15,9,30,00)
    for i in msft_cursor:
        if i['time'] >= time:
            msft_data.append(float(i['price'].replace(',','')))


    msft_data = np.array(msft_data).reshape(-1, 1)
    train_x = msft_data[:training_size]
    train_y = msft_data[1:training_size + 1]
    test_x = msft_data[training_size:]
    test_y = msft_data[training_size + 1:]

    test_x = test_x[0].reshape(-1,1)
    return train_x,train_y,test_x,test_y

def combine_data(name):
    data_y = db[name]
    y_cursor = data_y.find({})
    set_y = []
    for i in y_cursor:
        set_y.append(float(i['price'].replace(',','')))
    set_y = np.array(set_y).reshape(-1,1)
    train_y = set_y[1:training_size+1]
    test_y = set_y[training_size+1:]

    train_x = []
    test_x = []
    for st in stock_list:
        tmp_trx, tmp_try,tmp_tex,tmp_tey = collect_data(st)
        train_x.append(tmp_trx.reshape(-1,training_size))
        test_x.append(tmp_tex)
    print(train_x)
    train_x = np.array(train_x).reshape(5,12000).T
    test_x = np.array(test_x).reshape(-1,5)
    print(test_x.shape)
    print(test_x)
    return train_x,train_y, test_x, test_y

def svm_model_test(train_x,train_y,test_x,test_y):
    test_size = len(test_y)
    clf = svm.SVR()
    clf.fit(train_x, train_y)

    result = []
    tmp = test_x
    for i in range(test_size):
        price = clf.predict(tmp.reshape(-1, 1))
        result.append(price)
        tmp = price

    print(mean_squared_error(result, test_y))
    plt.title('SVM (06/11/2020)')
    plt.xlabel('last minute before closing ')
    plt.ylabel('price')
    plt.ticklabel_format(useOffset=False)
    plt.xticks(np.arange(0,test_size * 5,5))
    time = [i for i in range(0,test_size * 5,5)]
    plt.plot(time, test_y, 'b', label = 'test')
    plt.plot(time,result, 'r', label = 'prediction')
    plt.legend()
    plt.savefig('SVM.png')

    plt.show()


def etn_reg(train_x,train_y,test_x,test_y):
    test_size = len(test_y)
    etn = ElasticNet()
    etn.fit(train_x, train_y)

    result = []
    tmp = test_x
    for i in range(test_size):
        price = etn.predict(tmp)
        result.append(price)
        tmp = price.reshape(-1,1)
    print(mean_squared_error(result, test_y))
    plt.title('elastic net regression (06/11/2020)')
    plt.xlabel('last minute before closing ')
    plt.ylabel('price')
    time = [i for i in range(0, test_size * 5, 5)]
    plt.plot(time, test_y, 'b')
    plt.plot(time, result, 'r')
    plt.savefig('ETN.png')

    plt.show()


def lasso_reg(train_x,train_y,test_x,test_y):
    test_size = len(test_y)
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(train_x,train_y)

    result = []
    tmp = test_x
    for i in range(test_size):
        price = lasso.predict(tmp.reshape(-1, 1))
        result.append(price)
        tmp = price
    print(mean_squared_error(result, test_y))
    plt.title('lasso regression (06/11/2020)')
    plt.xlabel('last minute before closing ')
    plt.ylabel('price')
    plt.xticks(np.arange(0,test_size * 5,5))
    time = [i for i in range(0, test_size * 5, 5)]
    plt.plot(time, test_y, 'b', label = 'test')
    plt.plot(time, result, 'r', label = 'prediction')
    plt.legend()
    plt.savefig('lasso.png')

    plt.show()





if __name__ == '__main__':
    train_x, train_y, test_x, test_y = collect_data('FB')
    # train_x, train_y, test_x, test_y = combine_data('NVDA')
    # etn_reg(train_x,train_y,test_x,test_y)
    # svm_model_test(train_x,train_y,test_x,test_y)
    # lasso_reg(train_x, train_y, test_x, test_y)
