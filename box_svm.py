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

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.svm import SVC

training_size = 45000
max_error = 0.1



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

def feature_extract(segments):
    result = []
    for seg in segments:
        tmp = []
        interval = float(seg[2]) - float(seg[0])
        spread = abs(float(seg[1]) - float(seg[3]))
        label = 0
        if float(seg[1]) - float(seg[3]) >= 0:
            label = 0
        else:
            label = 1
        start_price = float(seg[1])
        end_price = float(seg[3])
        tmp.append(interval)
        tmp.append(spread)
        tmp.append(start_price)
        tmp.append(end_price)
        tmp.append(label)
        result.append(tmp)
    return np.array(result)

if __name__ == '__main__':
    client = MongoClient(port=27017)
    print('Connect MongoDB Successful')
    db = client.StockAnalyze
    names = ['NVDA','AAPL', 'FB','MSFT','TSLA']
    for name in names:

        msft = db[name]
        cursor = msft.find({})

        data = []
        for i in cursor:
            data.append(float(i['price'].replace(',', '')))

        data = np.array(data)
        time = np.array([i for i in range(len(data))])

        print(data.shape[0])

        segments = segment.topdownsegment(data,fit.interpolate,fit.sumsquared_error,max_error)
        input = feature_extract(segments)
        output = extract_new_y_feature(segments)
        print(input.shape, output.shape)
        size = int(input.shape[0]* 0.95)
    # train_x, train_y, test_x, test_y = train_test_split(input, output, test_size=0.9)
        train_x = input[:size]
        train_y = output[1:size+1]
        test_x = input[size:-1]
        test_y = output[size+1:]
    # segments_train = segment.topdownsegment(train_x, fit.interpolate, fit.sumsquared_error, max_error)
    # segments_test = segment.topdownsegment(test_x, fit.interpolate, fit.sumsquared_error, max_error)
    # train_y = extract_new_y_feature(segments_train)
    # test_y = extract_new_y_feature(segments_test)
    # train_x = feature_extract(segments_train)
    # test_x = feature_extract(segments_test)
    # test_y.reshape(-1,1)
    # train_y.reshape(-1,1)
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    # train_y = np.append(train_y[1:], test_y[0], axis=0)
    # test_x = test_x[:-1]
    # test_y = test_y[1:]
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # train_y = add_label(segments_train)
    # test_x = add_label(segments_test)
    # print(train_x.shape)
    # print(train_y.shape)
    # print(test_x.shape)
    # print(test_y.shape)
    # train_x, test_x = extract_new_x_feature(segments_train, segments_test)
    # train_y = extract_new_y_feature(segments_train)
    # test_y = extract_new_y_feature(segments_test)
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
                             'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [100]},
                            {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000],'max_iter': [100]}]

        clf = GridSearchCV(SVC(), tuned_parameters)
        clf.fit(train_x,train_y)
        print('----------' + name + '----------')
        print('fit finished')
        print(clf.best_params_)
        result = clf.predict(test_x)
        plt.title(name)
        plt.plot(result, 'r', label = 'predict')
        plt.plot(test_y,'b', label = 'real')
        plt.xlabel('time')
        plt.ylabel('price')
        plt.legend()
        fig_name = name + '.pdf'
        plt.savefig(fig_name)
        plt.show()


        print(classification_report(test_y, result))
        print(accuracy_score(test_y, result))
        print(f1_score(test_y, result))
        print(recall_score(test_y,result))
        print(precision_score(test_y,result))
        print('----------------------------')





