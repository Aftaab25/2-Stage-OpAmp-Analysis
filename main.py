import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Header for the main Content
st.header('Two Stage Operational Amplifier')

# Header for the sidebar
st.sidebar.header('User Input Features')

df = pd.read_csv('2STAGEOPAMP_DATASET.csv')
st.subheader('Description of the Dataset')
columns = ['Is4','Gm6','Gm4','Asp_1','Asp_2','Asp_3','Asp_4','Asp_5','Abs_Gain','Delay']
df.drop(columns,axis='columns',inplace=True)

st.write(df.describe())

st.write(df)


# Working with the Neural Network

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

model = load_model('model.h5')

dc = st.text_input('Pick a number')
st.text_input('Pick a number')
st.text_input('Pick a number')
st.text_input('Pick a number')
st.text_input('Pick a number')

st.subheader('Working With the Neural Network')

arr = np.array([[0.0000, 0.0000, 0.0001, 0.0001, 20.0438]])

predictions = model.predict(arr)
# predictions = model.predict(arr)
st.write("Predicted Values are:",predictions)
st.write("Real Values are:",y_test[:5])
st.write("Aspect Ratios are:",predictions*2)

y_pred = model.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))

mae_neural,mse_neural = model.evaluate(X_test_scaled,y_test)
st.write("Mean Absolute Error:",mae_neural)
st.write("Root Mean Squared Error:",mse_neural**0.5)
st.write("Mean Squared Error:",mse_neural)

st.write('R^2 Score :%.3f' % r2_score(y_test,y_pred))

r_squared = r2_score(y_test,y_pred)
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
st.write('Adjusted R^Score : .%3f' %adjusted_r_squared)

print(type(X_test_scaled[:1]) == type(arr))


