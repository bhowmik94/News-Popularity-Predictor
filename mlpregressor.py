from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import metrics
df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.drop('url', 1)
df = df.drop(' timedelta',1)
#df = df.drop(' shares',1)
X = df[df.columns[0:58]]
y = df[df.columns[-1]]

X = np.asarray(X)
y = np.asarray(y)
miny = min(y)
maxy = max(y)
#print(miny)
#print(maxy)

sclr = preprocessing.MinMaxScaler()
#X_train = X
#y_train = y
X_train = sclr.fit_transform(X)
y_train = sclr.fit_transform(y)
scale = (maxy-miny)
#X_test = sclr.fit_transform(test_x)
#y_test = sclr.fit_transform(test_y)
num = 25000
tr_x = X_train[0:num]
tr_y = y_train[0:num]

test_x = X_train[num:]
test_y = y_train[num:]
#print(y_test)

clf = MLPRegressor(hidden_layer_sizes=(10,10,),
                   activation='relu',
                   solver = 'adam',
                   learning_rate = 'constant',
                   learning_rate_init=0.05,
                   verbose=True)
clf.fit(tr_x,tr_y)
y_pred = clf.predict(test_x)
#print( clf.score(val_x,val_y), df.shape, " ", x.shape, " ", y.shape)
#print(clf.predict(x[39643]))
"""
y_pr = []
y_ts = []
for i in range(len(y_pred)):
    temp1 = y_pred[i]
    temp2 = test_y[i]
    y_pr.append(temp1)
    y_ts.append(temp2)
"""
mse = metrics.mean_squared_error(test_y,y_pred)*scale
#print(metrics.mean_squared_error(test_y,y_pred),"\t",mse,"\t",np.sqrt(mse))
print("RMSE: ",np.sqrt(mse))
