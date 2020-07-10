import tensorflow as tf
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
from pymongo import MongoClient
import segment
import fit
from tensorflow import keras

max_error = 0.1

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

    return np.array(label)


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

        print(data.shape)

        segments = segment.topdownsegment(data, fit.interpolate, fit.sumsquared_error, max_error)
        input = feature_extract(segments)
        output = extract_new_y_feature(segments)
        print(input.shape, output.shape)
        size = int(input.shape[0]* 0.95)

        train_x = input[:size]
        train_y = output[1:size+1]
        test_x = input[size:-1]
        test_y = output[size+1:]
        train_size = train_x.shape[0]
        test_size = test_y.shape[0]
        model = model_defination()
        train_x = train_x.reshape((train_size, 5, 1, 1))
        history = model.fit(train_x, train_y, epochs=30, validation_data=(train_x, train_y))
        test_x = test_x.reshape((test_size, 5, 1, 1))
        model.evaluate(test_x, test_y, verbose=1)