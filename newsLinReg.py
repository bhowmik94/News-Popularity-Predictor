import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#from sklearn.cross_validation import train_test_split
df = pd.read_csv('OnlineNewsPopularity.csv')
X = df[df.columns[2:60]]
y = df[df.columns[-1]]
X = np.asarray(X)
y = np.asarray(y)
X_train = X[0:39000]
y_train = y[0:39000]
X_test = X[39000:]
y_test = y[39000:]
#X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
linReg = LinearRegression()
linReg.fit(X_train,y_train)
y_pred = linReg.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print(y_pred[-1],"\t",y_test[-1])