import warnings
import logging
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from docopt import docopt
n_steps_frac_change=50
n_steps_frac_high=10
n_steps_frac_low=10
n_latency_days=10

def feature_extract(data):
    open_price = np.array(data['Open'])
    close_price = np.array(data['Close'])
    high_price = np.array(data['High'])
    low_price = np.array(data['Low'])

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
    previous_data = test_data.iloc[previous_data_end_index:previous_data_start_index]
    previous_data_feature = feature_extract(previous_data)
    outcome_score = []

    for po in possibile_outcomes:
        total_data = np.row_stack((previous_data_feature, po))
        outcome_score.append(hmm.score(total_data))

    most_probable_outcome = possibile_outcomes[np.argmax(outcome_score)]
    return most_probable_outcome

def predict_close_price(day_index):
    open_price = test_data.iloc[day_index]['Open']
    predicted_frac_change, _,_ = get_most_probable_outcome(day_index)

    return open_price * (1 + predicted_frac_change)

def predict_close_price_for_days(days, name):
    predicted_close_prices = []
    for day_index in tqdm(range(days)):
        predicted_close_prices.append(predict_close_price(day_index))

    test = test_data[0:days]
    days = np.array(test['Date'], dtype="datetime64[ms]")
    actual_close_prices = test['Close']
    plt.xlabel('time series')
    plt.ylabel('close price')
    plt.plot(days, actual_close_prices, 'b', label = 'actual')
    plt.plot(days, predicted_close_prices, 'r', label = 'predicted')
    plt.title(name)


    plt.legend()
    plt.savefig('historical_'+ name +'.png')
    plt.show()

    return predicted_close_prices

def model_run(name):
    data = pd.read_csv('data/' + name + '.csv')
    train_data, test_data = train_test_split(data, test_size=0.33, shuffle=False)
    feature_vector = feature_extract(train_data)
    print(feature_vector)

    hmm = GaussianHMM(n_components=4)
    hmm.fit(feature_vector)

    possibile_outcomes = compute_all_possible_outcomes(n_steps_frac_low, n_steps_frac_high, n_steps_frac_change)
    predict = predict_close_price_for_days(500, name)

# for stock in stocks:
#     data = pd.read_csv('data/'+ stock +'.csv')
#     train_data, test_data = train_test_split(data, test_size=0.33, shuffle=False)
#     feature_vector = feature_extract(train_data)
#     print(feature_vector)
#
#     hmm = GaussianHMM(n_components=4)
#     hmm.fit(feature_vector)
#
#     possibile_outcomes = compute_all_possible_outcomes(n_steps_frac_low,n_steps_frac_high,n_steps_frac_change)
#     predict = predict_close_price_for_days(500, stock)