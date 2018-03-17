from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from math import sqrt
import  pandas as pd
import numpy as np


filename = 'task1a_lm1d1z/train.csv'
data = pd.read_csv(filename)

y = data['y']
X = data.drop(['Id','y'],axis=1)

lam = [0.1, 1, 10, 100, 1000]
ridge = Ridge(normalize=False)
rms=[]
for parameter in lam:
    ridge.alpha = parameter
    predicted = cross_val_predict(ridge,X, y, cv=10)
    rms.append(np.mean(mean_squared_error(predicted,y)))

print (rms)
