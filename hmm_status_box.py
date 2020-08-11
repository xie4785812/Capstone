import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pymongo import MongoClient
import segment
import datetime
import fit

n_steps_frac_change=50
n_steps_frac_high=10
n_steps_frac_low=10
n_latency_days=10

client = MongoClient(port = 27017)
client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze
max_error = 0.1
stocks = ['NVDA', 'MSFT','AAPL', 'FB', 'TSLA']


def build_box_feature(segments, data):
    result = []
    for seg in segments:
        box_feature = []
        start_time = int(seg[0])
        end_time = int(seg[2])
        start_price = float(seg[1])
        end_price = float(seg[3])
        tmp_set = data[start_time: end_time]
        box_feature.append(start_price)
        box_feature.append(end_price)
        box_feature.append(max(tmp_set))
        box_feature.append(min(tmp_set))
        result.append(box_feature)
    return np.array(result)




def feature_extract(data):
    open_price = np.array(data[:, 0])
    close_price = np.array(data[:, 1])
    high_price = np.array(data[:, 2])
    low_price = np.array(data[:, 3])

    frac_change = (close_price - open_price) / open_price
    frac_high = (high_price - open_price) / open_price
    frac_low = (open_price - low_price) / open_price
    return np.column_stack((frac_change, frac_high, frac_low))



def compute_all_possible_outcomes(n_steps_frac_change, n_steps_frac_high,n_steps_frac_low):
    frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
    frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
    frac_low_range = np.linspace(0,0.1, n_steps_frac_low)

    return np.array(list(itertools.product(frac_change_range,frac_high_range,frac_low_range)))



def get_most_probable_outcome(day_index):
    previous_data_start_index = max(0, day_index - n_latency_days)
    previous_data_end_index = max(0, day_index - 1)
    previous_data = test_data[previous_data_end_index:previous_data_start_index]
    previous_data_feature = feature_extract(previous_data)
    outcome_score = []

    for po in possibile_outcomes:
        total_data = np.row_stack((previous_data_feature, po))
        outcome_score.append(hmm.score(total_data))

    most_probable_outcome = possibile_outcomes[np.argmax(outcome_score)]
    return most_probable_outcome

def predict_close_price(day_index):
    open_price = test_data[day_index,0]
    predicted_frac_change, _,_ = get_most_probable_outcome(day_index)

    return open_price * (1 + predicted_frac_change)

def predict_close_price_for_days(days, stock_name):
    predicted_close_prices = []
    for day_index in tqdm(range(days)):
        predicted_close_prices.append(predict_close_price(day_index))

    test = test_data[0:days]
    # days = np.array(test['Date'], dtype="datetime64[ms]")
    actual_close_prices = test[:,1]

    plt.plot(actual_close_prices, 'b', label = 'actual')
    plt.plot(predicted_close_prices, 'r', label = 'predicted')
    plt.title(stock_name)


    plt.legend()
    name = stock_name + '_box.png'
    plt.savefig(name)
    plt.show()

    return predicted_close_prices

def model_run(name):
    msft = db[name]
    msft_cursor = msft.find({})

    data = []
    time = datetime.datetime(2020, 6, 15, 9, 30, 00)
    for i in msft_cursor:
        if i['time'] >= time:
            data.append(float(i['price'].replace(',', '')))

    data = np.array(data)
    segments = segment.topdownsegment(data, fit.interpolate, fit.sumsquared_error, max_error)
    box_feature = build_box_feature(segments, data)
    print(box_feature)
    train_data, test_data = train_test_split(box_feature, test_size=0.33, shuffle=False)
    feature_vector = feature_extract(train_data)
    print(feature_vector)

    hmm = GaussianHMM(n_components=4)
    hmm.fit(feature_vector)
    possibile_outcomes = compute_all_possible_outcomes(n_steps_frac_low, n_steps_frac_high, n_steps_frac_change)
    predict = predict_close_price_for_days(500, name)
# for stock in stocks:
#     msft = db[stock]
#     msft_cursor = msft.find({})
#
#     data = []
#     time = datetime.datetime(2020, 6, 15, 9, 30, 00)
#     for i in msft_cursor:
#         if i['time'] >= time:
#             data.append(float(i['price'].replace(',', '')))
#
#     data = np.array(data)
#     segments = segment.topdownsegment(data, fit.interpolate, fit.sumsquared_error, max_error)
#     box_feature = build_box_feature(segments, data)
#     print(box_feature)
#     train_data, test_data = train_test_split(box_feature, test_size=0.33, shuffle=False)
#     feature_vector = feature_extract(train_data)
#     print(feature_vector)
#
#     hmm = GaussianHMM(n_components=4)
#     hmm.fit(feature_vector)
#     possibile_outcomes = compute_all_possible_outcomes(n_steps_frac_low, n_steps_frac_high, n_steps_frac_change)
#     predict = predict_close_price_for_days(500,stock)