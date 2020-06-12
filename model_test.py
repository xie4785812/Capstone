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
client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze

def collect_data(name):
    msft = db[name]

    msft_cursor = msft.find({})

    msft_data = []
    for i in msft_cursor:
        msft_data.append(float(i['price'].replace(',','')))

    msft_data = np.array(msft_data).reshape(-1, 1)
    train_x = msft_data[:-13]
    train_y = msft_data[1:-12]
    test_x = msft_data[-13:]
    test_y = msft_data[-12:]

    return train_x,train_y,test_x,test_y



def svm_model_test(train_x,train_y,test_x,test_y):
    clf = svm.SVR()
    clf.fit(train_x, train_y)

    result = []
    tmp = test_x[0:1].reshape(-1, 1)
    for i in range(12):
        price = clf.predict(tmp.reshape(-1, 1))
        result.append(price)
        tmp = price

    print(mean_squared_error(result, test_y))
    plt.title('SVM (06/11/2020)')
    plt.xlabel('last minute before closing ')
    plt.ylabel('price')
    plt.ticklabel_format(useOffset=False)
    plt.xticks(np.arange(0,60,5))
    time = [i for i in range(0,60,5)]
    plt.plot(time, test_y, 'b', label = 'test')
    plt.plot(time,result, 'r', label = 'prediction')
    plt.legend()
    plt.show()
    plt.savefig('SVM.png')

def etn_reg(train_x,train_y,test_x,test_y):
    etn = ElasticNet()
    etn.fit(train_x, train_y)

    result = []
    tmp = test_x[0:1].reshape(-1,1)
    for i in range(12):
        price = etn.predict(tmp.reshape(-1,1))
        result.append(price)
        tmp = price
    print(mean_squared_error(result, test_y))
    plt.title('elastic net regression (06/11/2020)')
    plt.xlabel('last minute before closing ')
    plt.ylabel('price')
    time = [i for i in range(0, 60, 5)]
    plt.plot(time, test_y, 'b')
    plt.plot(time, result, 'r')
    plt.show()
    plt.savefig('ETN.png')

def lasso_reg(train_x,train_y,test_x,test_y):
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(train_x,train_y)

    result = []
    tmp = test_x[0:1].reshape(-1, 1)
    for i in range(12):
        price = lasso.predict(tmp.reshape(-1, 1))
        result.append(price)
        tmp = price
    print(mean_squared_error(result, test_y))
    plt.title('lasso regression (06/11/2020)')
    plt.xlabel('last minute before closing ')
    plt.ylabel('price')
    plt.xticks(np.arange(0,60,5))
    time = [i for i in range(0, 60, 5)]
    plt.plot(time, test_y, 'b', label = 'test')
    plt.plot(time, result, 'r', label = 'prediction')
    plt.legend()
    plt.show()
    plt.savefig('lasso')




if __name__ == '__main__':
    train_x, train_y, test_x, test_y = collect_data('NVDA')
    etn_reg(train_x,train_y,test_x,test_y)
    svm_model_test(train_x,train_y,test_x,test_y)
    lasso_reg(train_x, train_y, test_x, test_y)
