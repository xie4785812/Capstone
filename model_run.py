import numpy as np
from sklearn.linear_model import ElasticNet,Lasso
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.svm import SVC,SVR
from sklearn import svm
import tensorflow as tf
import matplotlib.pyplot as plt
from pymongo import MongoClient
import segment
import fit
import sys
import model_test
import datetime
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import hmm
import hmm_status_box

n_steps_frac_change=50
n_steps_frac_high=10
n_steps_frac_low=10
n_latency_days=10

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

def model_defination():
    """
    model definition
    :return: model
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(2, (2, 1), padding='same', activation='relu', input_shape=(5, 1, 1)),
        tf.keras.layers.MaxPool2D((2, 1)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Conv2D(6, (2, 1), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def seg_extract(data_name):
    client = MongoClient(port=27017)
    print('Connect MongoDB Successful')
    db = client.StockAnalyze
    msft = db[data_name]
    cursor = msft.find({})
    data = []
    for i in cursor:
        data.append(float(i['price'].replace(',', '')))
    data = np.array(data)
    segments = segment.topdownsegment(data, fit.interpolate, fit.sumsquared_error, max_error)

    input = feature_extract(segments)
    output = extract_new_y_feature(segments)

    size = int(input.shape[0] * 0.95)

    train_x = input[:size]
    train_y = output[1:size + 1]
    test_x = input[size:-1]
    test_y = output[size + 1:]

    return train_x,train_y,test_x,test_y

def svm_box(train_x,train_y,test_x,test_y):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [100]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [100]}]

    clf = GridSearchCV(SVC(), tuned_parameters)
    clf.fit(train_x, train_y)
    print('----------' + data_select + '----------')
    print('fit finished')
    print(clf.best_params_)
    result = clf.predict(test_x)
    plt.title(data_select)
    plt.plot(result, 'r', label='predict')
    plt.plot(test_y, 'b', label='real')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend()
    fig_name = data_select + '.pdf'
    plt.savefig(fig_name)
    plt.show()

    print(classification_report(test_y, result))
    print('Accuracy Score: ', accuracy_score(test_y, result))
    print('F1 Score:', f1_score(test_y, result))
    print('Recall Score:', recall_score(test_y, result))
    print('Precision Score:', precision_score(test_y, result))

def bpn_box(train_x, train_y, test_x, test_y):
    train_size = train_x.shape[0]
    test_size = test_y.shape[0]
    model = model_defination()
    train_x = train_x.reshape((train_size, 5, 1, 1))
    history = model.fit(train_x, train_y, epochs=30, validation_data=(train_x, train_y))
    test_x = test_x.reshape((test_size, 5, 1, 1))
    model.evaluate(test_x, test_y, verbose=1)
if __name__ == '__main__':

    names = ['NVDA','AAPL', 'FB','MSFT','TSLA']
    data_select = sys.argv[1]
    if data_select in names:
        client = MongoClient(port=27017)
        print('Connect MongoDB Successful')
        db = client.StockAnalyze
        feature_select = sys.argv[2]
        if feature_select == 'simple':
            print('simple extraction regression with ' + data_select)
            model_run(data_select)
        elif feature_select == 'fix':
            print('Feature Extraction')
            train_x, train_y, test_x, test_y = seg_extract(data_select)
            model_select = sys.argv[3]
            if model_select == 'svm':
                print('start SVM')
                svm_box(train_x, train_y, test_x, test_y)
            elif model_select == 'bpn':
                print('start BPN')
                bpn_box(train_x, train_y, test_x, test_y)
            elif model_select == 'hmm':
                print('start HMM on historical data')
                hmm.model_run(data_select)
            elif model_select == 'hmm_box':
                print('start HMM on status box data')
                hmm_status_box.model_run(data_select)
            else:
                print('wrong model')
                sys.exit()
        else:
            print('wrong feature')
            sys.exit()
    elif data_select == 'run':
        client = MongoClient(port=27017)
        print('Connect MongoDB Successful')
        db = client.StockAnalyze

        for name in names:
            print('----------------------------------')
            print('simple extraction regression with ' + name)
            model_run(name)
            print('----------------------------------')
            print('Status Box Feature Extraction with ', name)
            train_x, train_y, test_x, test_y = seg_extract(name)
            print('Status Box Feature Extraction Done with ', name)
            print('----------------------------------')
            print('Status Box SVM with ', name)
            svm_box(train_x, train_y, test_x, test_y)
            print('Status Box SVM Done with ', name)
            print('----------------------------------')
            print('Status Box BPN with ', name)
            bpn_box(train_x, train_y, test_x, test_y)
            print('Status Box BPN Done with ', name)
            print('----------------------------------')
            print('Historical data HMM with ', name)
            hmm.model_run(name)
            print('Historical data HMM done with ', name)
            print('----------------------------------')
            print('Status box data HMM with ', name)
            hmm_status_box.model_run(name)
            print('Status box data HMM done with ', name)
            print('----------------------------------')



