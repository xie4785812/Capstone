from pymongo import MongoClient
import numpy as np
import sklearn
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

client = MongoClient(port=27017)
print('Connect MongoDB Successful')
db = client.StockAnalyze

database = db['NVDA']
gold = db['GOLD']

aggregate_1 = [{'$lookup':{'from':'Gold', 'localField':'date', 'foreignField':'date', 'as':'gold'}}]
output = database.aggregate(aggregate_1)
X = []
Y = []
for i in output:
    tmp = []
    tmp.append(i['open price'])
    tmp.append(i['gold'][0]['open price'])
    X.append(tmp)
    Y.append(i['close price'])




X = np.array(X)

Y = np.array(Y)

print(X.shape)

# train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=False)
# train_x = train_x.reshape(-1,1)
# train_y = train_y.reshape(-1,1)
# test_x = test_x.reshape(-1,1)
# test_y = test_y.reshape(-1,1)


etn = ElasticNet()

etn.fit(X,Y)

today_open = np.array([353.33, 24.65]).reshape(-1,2)
print(today_open.shape)
predict_test_elastic = etn.predict(today_open)
print(predict_test_elastic)

print(mean_squared_error([352.25], predict_test_elastic))

# lr = LinearRegression()
# lr.fit(X,Y)
# today_open = np.array([353.33, 24.65]).reshape(-1,2)
# pre = lr.predict(today_open)
# print(pre)
