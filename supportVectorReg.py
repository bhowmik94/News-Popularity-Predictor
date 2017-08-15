from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import metrics
df = pd.read_csv('output.csv')
df = df.drop('url', 1)
df = df.drop(' timedelta',1)
x = df[df.columns[0:58]]
y = df[df.columns[-1]]

x = np.asarray(x)
y = np.asarray(y)
length = len(y);
num=35000
tr_x = x[0:num]
tr_y = y[0:num]

val_x = x[num:]
val_y = y[num:]

clf = svm.SVR()
clf.fit(tr_x, tr_y)
y_pred = clf.predict(val_x)
#print( clf.score(val_x,val_y), df.shape, " ", x.shape, " ", y.shape)
print(np.sqrt(metrics.mean_squared_error(val_y,y_pred)))
print(y_pred[-1],"\t",val_y[-1])
#print(clf.predict(x[39643]))