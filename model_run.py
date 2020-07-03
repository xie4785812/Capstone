import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn import svm
import tensorflow as tf
import matplotlib.pyplot as plt
from pymongo import MongoClient
import segment
import fit
import sys

def collect_data(name):
    msft = db[name]
    msft_cursor = msft.find({})

    msft_data = []
    for i in msft_cursor:
        msft_data.append(float(i['price'].replace(',','')))

    msft_data = np.array(msft_data).reshape(-1, 1)
    training_size = msft_data.shape[0] * 0.95
    train_x = msft_data[:training_size]
    train_y = msft_data[1:training_size + 1]
    test_x = msft_data[training_size:]
    test_y = msft_data[training_size + 1:]

    test_x = test_x[0].reshape(-1,1)
    return train_x,train_y,test_x,test_y

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


if __name__ == '__main__':

    names = ['NVDA','AAPL', 'FB','MSFT','TSLA']
    data_select = sys.argv[1]
    if data_select in names:
        client = MongoClient(port=27017)
        print('Connect MongoDB Successful')
        db = client.StockAnalyze

        feature_select = sys.argv[2]

        if feature_select == 'simple':
            train_x, train_y, test_x, test_y = collect_data(data_select)
            model_select = sys.argv[3]
            if model_select == 'svm':
                print('Start SVM model')
                svm_model_test(train_x, train_y, test_x, test_y)
            elif model_select == 'lasso':
                print('Start Lasso Regression')
                lasso_reg(train_x, train_y, test_x, test_y)
            elif model_select == 'elastic':
                print('Start Elastic Net Model')
                etn_reg(train_x, train_y, test_x, test_y)
            else:
                print('wrong model')
                sys.exit()
        elif feature_select == 'fix':
            print('Feature Extraction')
            msft = db[data_select]
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

            model_select = sys.argv[3]
            if model_select == 'svm':
                print('start SVM')
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
            elif model_select == 'bpn':
                print('start BPN')
                train_size = train_x.shape[0]
                test_size = test_y.shape[0]
                model = model_defination()
                train_x = train_x.reshape((train_size, 5, 1, 1))
                history = model.fit(train_x, train_y, epochs=30, validation_data=(train_x, train_y))
                test_x = test_x.reshape((test_size, 5, 1, 1))
                model.evaluate(test_x, test_y, verbose=1)
            elif model_select == 'adaboostSVM':
                print(8)
            else:
                print('wrong model')
                sys.exit()
        else:
            print('wrong feature')
            sys.exit()
