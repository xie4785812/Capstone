from pymongo import MongoClient
import numpy as np
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet,Lasso
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import svm
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

import warnings

warnings.filterwarnings('ignore')

client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze

training_size = 80000
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
    print(test_x)
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
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [100]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [100]}]

    clf = GridSearchCV(SVR(), tuned_parameters)

    clf.fit(train_x, train_y)
    result = clf.predict(test_x)
    print(clf.best_params_)
    score = accuracy_cal(test_x,test_y,result)
    print(score)
    print('svm done')
    return result
    # print(mean_squared_error(result, test_y))
    # plt.title('SVM (06/11/2020)')
    # plt.xlabel('last minute before closing ')
    # plt.ylabel('price')
    # plt.ticklabel_format(useOffset=False)
    # plt.xticks(np.arange(0,test_size * 5,5))
    # time = [i for i in range(0,test_size * 5,5)]
    # plt.plot(time, test_y, 'b', label = 'test')
    # plt.plot(time,result, 'r', label = 'prediction')
    # plt.legend()
    # plt.savefig('SVM.png')
    #
    # plt.show()


def etn_reg(train_x,train_y,test_x,test_y):
    alpha = [0.001,0.01,0.1,1,10,100,1000]
    l1_ratio = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    param_grid = dict(alpha = alpha, l1_ratio = l1_ratio)
    etn = GridSearchCV(ElasticNet(), param_grid)
    etn.fit(train_x, train_y)
    print(etn.best_params_)
    result = etn.predict(test_x)
    score = accuracy_cal(test_x, test_y, result)
    print(score)
    print('etn done')
    return result
    # print(mean_squared_error(result, test_y))
    # plt.title('elastic net regression (06/11/2020)')
    # plt.xlabel('last minute before closing ')
    # plt.ylabel('price')
    # time = [i for i in range(0, test_size * 5, 5)]
    # plt.plot(time, test_y, 'b')
    # plt.plot(time, result, 'r')
    # plt.savefig('ETN.png')
    #
    # plt.show()


def lasso_reg(train_x,train_y,test_x,test_y):
    test_size = len(test_y)
    tuned_parameters = {'alpha':[5,10,20]}
    lasso = GridSearchCV(Lasso(), tuned_parameters)
    lasso.fit(train_x,train_y)
    print(lasso.best_params_)
    result = lasso.predict(test_x)
    score = accuracy_cal(test_x,test_y,result)
    print(score)
    print('lasso done')
    return result

    # print(mean_squared_error(result, test_y))
    # plt.title('lasso regression (06/11/2020)')
    # plt.xlabel('last minute before closing ')
    # plt.ylabel('price')
    # plt.xticks(np.arange(0,test_size * 5,5))
    # time = [i for i in range(0, test_size * 5, 5)]
    # plt.plot(time, test_y, 'b', label = 'test')
    # plt.plot(time, result, 'r', label = 'prediction')
    # plt.legend()
    # plt.savefig('lasso.png')
    #
    # plt.show()


def model_run(name):
    train_x, train_y, test_x, test_y = feature_extraction(name)
    lasso_result = lasso_reg(train_x,train_y,test_x,test_y)
    svm_result = svm_model_test(train_x,train_y,test_x,test_y)
    etn_result = etn_reg(train_x,train_y,test_x,test_y)
    # plt.figure(figsize=(20,10),dpi=150)
    plt.title('Lasso vs ETN vs SVM of '+ name)
    plt.xlabel('last minute before closing')
    plt.ylabel('price')
    plt.plot(lasso_result[0:100], 'orange' ,label = 'LASSO Result')
    plt.plot(svm_result[0:100], 'red', label = 'SVM Result')
    plt.plot(etn_result[0:100], 'green', alpha=0.5, label = 'ETN Result')
    plt.plot(test_y[0:100], 'black', label = 'Actual Price')
    plt.legend()
    plt.savefig('svm_etn_lasso_'+ name + '.png')
    plt.show()


def feature_extraction(name):
    msft = db[name]
    msft_cursor = msft.find({})

    msft_data = []
    time = datetime.datetime(2020, 6, 15, 9, 30, 00)
    for i in msft_cursor:
        if i['time'] >= time:
            msft_data.append(float(i['price'].replace(',', '')))
    msft_data = np.array(msft_data).reshape(-1, 1)
    train_data, test_data = train_test_split(msft_data, test_size=0.2, shuffle=False)
    train_x = train_data[:-12]
    train_y = train_data[12:]
    test_x = test_data[:-12]
    test_y = test_data[12:]
    return train_x,train_y,test_x,test_y

def accuracy_cal(test_x, test_y, pred_y):
    total = 0
    right = 0
    for i in range(len(test_x)):
        total += 1
        if test_y[i] - test_x[i] > 0 and pred_y[i] - test_x[i] > 0:
            right+=1
        elif test_y[i] - test_x[i] < 0 and pred_y[i] - test_x[i] < 0:
            right+=1
    return right/total
if __name__ == '__main__':

    for i in stock_list:
        print(i)
        model_run(i)
    # train_x, train_y, test_x, test_y = combine_data('NVDA')
    # etn_reg(train_x,train_y,test_x,test_y)
    # svm_model_test(train_x,train_y,test_x,test_y)
    # lasso_reg(train_x, train_y, test_x, test_y)\

