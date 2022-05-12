# Importing Necessary Libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.layers import LeakyReLU, PReLU, ELU
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
import tensorflow as tf
from keras.layers import BatchNormalization

import pickle

df = pd.read_csv('2STAGEOPAMP_DATASET.csv')

columns = ['Is4','Gm6','Gm4','Asp_1','Asp_2','Asp_3','Asp_4','Asp_5','Abs_Gain','Delay']
df.drop(columns,axis='columns',inplace=True)

# print(df.describe())

# Split the Features and Labels and dividing into training and testing data
labels = ['Wi1', 'Wi2', 'Wi3', 'Wi4', 'Wi5']
features = ['DC Gain', 'ft', 'f3', 'Pdiss', 'Vcm']
X = df.drop(labels, axis=1)
y = df.drop(features, axis=1)
# print(x)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)


scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train*(10**6)
y_test = y_test*(10**6)

# Linear Regression
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.005)
ridge_reg.fit(X_train_scaled,y_train)

y_pred_ridge = ridge_reg.predict(X_test_scaled)


mse_ridge_reg = mean_squared_error(y_test,y_pred_ridge)
mae_ridge_reg = mean_absolute_error(y_test,y_pred_ridge)
print("Mean Squared Error:",mse_ridge_reg)
print("Root Mean Squared Error:",mse_ridge_reg**0.5)
print("Mean Absolute Error:",mae_ridge_reg)
from sklearn.metrics import r2_score
print('R^2 Score :%.3f' % r2_score(y_test,y_pred_ridge))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_ridge, c='black')
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.show()