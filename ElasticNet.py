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



train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.01, random_state=0, shuffle=False)
def predict_in_days(train_x, train_y, test_x, test_y):
    etn = ElasticNet()
    etn.fit(train_x, train_y)
    result = []
    begin_test = test_x[0]
    print(begin_test)
    for i in range(len(test_x)-1):
        tmp_closing = etn.predict(begin_test.reshape(-1,2))
        print(tmp_closing)
        result.append(tmp_closing)
        print(result)
        begin_test[0] = tmp_closing
        begin_test[1] = test_x[i+1,1]
        print(begin_test)
    return np.array(result)

predice_test_elastic = predict_in_days(train_x,train_y,test_x,test_y)
#
# print(mean_squared_error(test_y, predice_test_elastic))

plt.title('Elastic Net Prediction Curve')
plt.plot(test_y,'b')
plt.plot(predice_test_elastic,'r')
plt.show()



