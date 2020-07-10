import pwlf
import numpy as np
from pymongo import MongoClient
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show,legend,savefig
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import segment
import fit
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing

import pandas as pd
from sklearn.datasets import make_hastie_10_2

training_size = 68000
max_error = 0.1

client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze

msft = db['NVDA']
cursor = msft.find({})

data = []
for i in cursor:
    data.append(float(i['price'].replace(',','')))

data = np.array(data)
time = np.array([i for i in range(len(data))])



def add_label(segments):
    label = []
    tmp = 0
    for seg in segments:
        start = int(seg[0])
        end = int(seg[2])
        start_price = float(seg[1])
        end_price = float(seg[3])
        if start_price >= end_price:
            tmp = 0
        elif start_price <  end_price:
            tmp = 1

        for i in range(start, end):
            label.append(tmp)
    label.append(tmp)
    return np.array(label)

def error_rate(y,pred):
    return sum(y!= pred)/len(y)

def initclf(X_train,y_train,X_test,y_test,clf):
    clf.fit(X_train,y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_err = error_rate(y_train_pred,y_train)
    test_err = error_rate(y_test_pred,y_test)
    return train_err,test_err

def adaboost(X_train,y_train,X_test,y_test,M,clf):
    w = np.ones(len(X_train))/len(X_train)
    #刚开始总的分类器都是0
    n_train = len(X_train)
    n_test = len(y_train)
    pred_train,pred_test  = list(np.zeros(n_train)),list(np.zeros(n_test))
    for i in range(M):
        w1 = w*n_train
        clf.fit(X_train, y_train,sample_weight = w1)
        y_train_i = clf.predict(X_train)
        y_test_i = clf.predict(X_test)

        # miss is 8.1(b) 中的计算分类误差率要乘以w的
        miss = [int(i) for i in (y_train_i != y_train)]

        # miss2 是8.5中y*G(m)
        miss1 = [x if x == 1 else -1 for x in miss]
        #要注意np.dot()也可以一个ndarry 一个列表相乘 这里计算分类误差率和alpha_m
        error_m =np.dot(w,miss)
        print(error_m)
        alpha_m = 0.5 *np.log((1-error_m)/error_m)
        #更新权重
        w = np.multiply(w,np.exp([-alpha_m * x for x in miss1 ]))

        #ensemble
        pred_train = [sum(x) for x in zip(pred_train,[alpha_m * i for i in y_train_i ])]
        pred_test = [sum(x) for x in zip(pred_test,[alpha_m * i for i in y_test_i ])]
    pred_train,pred_test = np.sign(np.array(pred_train)),np.sign(np.array(pred_test))
    return error_rate(pred_train,y_train), error_rate(pred_test,y_test), y_test, pred_test

def plot_error_rate(er_train,er_test):
    df_err =pd.DataFrame([er_train,er_test]).T
    df_err.columns = ["Train","Test"]
    plot1 = df_err.plot(linewidth =3,figsize =(8,6),
                        color = ["lightblue","darkblue"],grid = True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')
    plt.show()


def collect_data(name):
    msft = db[name]
    msft_cursor = msft.find({})

    msft_data = []
    for i in msft_cursor:
        msft_data.append(float(i['price'].replace(',','')))



    msft_data = np.array(msft_data).reshape(-1, 1)
    train_x = msft_data[:training_size]
    train_y = msft_data[1:training_size + 1]
    test_x = msft_data[training_size:]
    test_y = msft_data[training_size + 1:]

    # test_x = test_x[0].reshape(-1,1)
    return train_x,train_y,test_x,test_y

def extract_new_x_feature(segements_train, segments_test):
    result_train = []
    for i in segements_train:
        tmp = []
        tmp.append(i[1])
        tmp.append(i[2]-i[0])
        result_train.append(tmp)

    result_test = []
    for i in segments_test:
        tmp = []
        tmp.append(i[1])
        tmp.append(i[2]-i[0])
        result_test.append(tmp)
    return np.array(result_train),np.array(result_test)

def extract_new_y_feature(segments):
    label = []
    tmp = 0
    for seg in segments:
        start = int(seg[0])
        end = int(seg[2])
        start_price = float(seg[1])
        end_price = float(seg[3])
        if start_price >= end_price:
            tmp = 0
        elif start_price <  end_price:
            tmp = 1

        label.append(tmp)

    return np.array(label)



if __name__ == '__main__':
    train_x, train_y, test_x, test_y = collect_data('AAPL')
    segments_train = segment.topdownsegment(train_x, fit.interpolate, fit.sumsquared_error, max_error)
    segments_test = segment.topdownsegment(test_x, fit.interpolate, fit.sumsquared_error, max_error)
    train_y = extract_new_y_feature(segments_train)
    test_y = extract_new_y_feature(segments_test)
    train_x,test_x = extract_new_x_feature(segments_train,segments_test)
    # train_x = preprocessing.normalize(train_x, axis=0)
    # test_x = preprocessing.normalize(test_x, axis = 0)
    print(train_x)
    print(train_x,train_y,test_x,test_y)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    svm_clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                          max_iter=-1, probability=False, random_state=None, shrinking=True,
                          tol=0.001, verbose=False)

    # er_train,er_test = initclf(train_x, train_y, test_x, test_y, svm_clf)
    # er_train, er_test = [er_train],[er_test]
    #
    # for i in range(1,20,1):
    #     print('step' + str(i))
    #     er_train_i,er_test_i,y_test,predict_test = adaboost(train_x,train_y,test_x,test_y,i,svm_clf)
    #     plt.title('status box real vs prediction')
    #     plt.xlabel('time series')
    #     plt.ylabel('status box (0 or 1)')
    #     plt.plot(y_test,'r', label = 'real')
    #     plt.plot(predict_test,'b',label = 'predict')
    #     plt.legend()
    #     plt.show()
    #     time = len(y_test)
    #     print(time, y_test)
    #     print(time, predict_test)
    #     er_train.append(er_train_i)
    #     er_test.append(er_test_i)
    # plot_error_rate(er_train,er_test)

    svm_clf.fit(train_x,train_y)
    result = svm_clf.predict(test_x)
    print(test_y)
    print(result)
    print('train_set: ', svm_clf.score(train_x,train_y))
    print('test set: ',svm_clf.score(test_x,test_y))







