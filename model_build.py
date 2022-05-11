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

print(df.describe())

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

# model = Sequential()
# model.add(Dense(units=3000,kernel_initializer='he_uniform',activation='relu',input_dim=5))
# #model.add(Dropout(0.4))
# #model.add(BatchNormalization())
# model.add(Dense(units=300,kernel_initializer='he_uniform',activation='relu'))
# #model.add(BatchNormalization())
# model.add(Dropout(0.01))
# model.add(Dense(units=30,kernel_initializer='he_uniform',activation='relu'))
# #model.add(BatchNormalization())
# model.add(Dropout(0.01))
# model.add(Dense(units=5,activation='linear'))
# model.compile(optimizer='Adam',loss='mean_absolute_error',metrics=['mse'])
# model.summary()

new_model = load_model('model.h5')

# history = new_model.fit(X_train_scaled,y_train,validation_split=0.15,batch_size=10,epochs=2000)



# pickle.dump(model, open('model.pkl', 'wb'))

# from matplotlib import pyplot as plt
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1,len(loss)+1)
# plt.plot(epochs,loss,'y',label='Training Loss')
# plt.plot(epochs,val_loss,'r',label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# #%%
# acc = history.history['mse']
# val_acc = history.history['val_mse']
# plt.plot(epochs,acc,'y',label='Training MSE')
# plt.plot(epochs,val_acc,'r',label='Validation MSE')
# plt.title('Training and Validation MSE')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


#%%
mae_neural,mse_neural = new_model.evaluate(X_test_scaled,y_test)
print("Mean Absolute Error:",mae_neural)
print("Root Mean Squared Error:",mse_neural**0.5)
print("Mean Squared Error:",mse_neural)

#%%
# Model Predictions
predictions = new_model.predict(X_test_scaled[:5])
print("Predicted Values are:",predictions)
print("Real Values are:",y_test[:5])
print("Aspect Ratios are:",predictions*2)

#%%
from sklearn.metrics import mean_squared_error
y_pred = new_model.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.metrics import r2_score
print('R^2 Score :%.3f' % r2_score(y_test,y_pred))
#%%
#plt.scatter(y_test,y_pred)
#%%
r_squared = r2_score(y_test,y_pred)
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print('Adjusted R^Score : .%3f' %adjusted_r_squared)