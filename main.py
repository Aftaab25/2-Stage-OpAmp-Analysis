import streamlit as st
import pyautogui
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
st.header('Aspect Ration Estimation of a Two-Stage Operational Amplifier')

# Reading the data
df = pd.read_csv('2STAGEOPAMP_DATASET.csv')
st.subheader('Description of the Dataset')
columns = ['Is4','Gm6','Gm4','Asp_1','Asp_2','Asp_3','Asp_4','Asp_5','Abs_Gain','Delay']
df.drop(columns,axis='columns',inplace=True)

# Displaying the description of Dataset
st.write(df.describe())

st.write(df)

# Split the Features and Labels and dividing into training and testing data
labels = ['Wi1', 'Wi2', 'Wi3', 'Wi4', 'Wi5']
features = ['DC Gain', 'ft', 'f3', 'Pdiss', 'Vcm']
X = df.drop(labels, axis=1)
y = df.drop(features, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train*(10**6)
y_test = y_test*(10**6)


def calculate(dc, ft, f3, Vcm, pdiss):
    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)
    st.write(test_user_scaled)
    # y_test = ans * (10 ** 6)
    model = load_model('model.h5')
    predictions = model.predict(test_user_scaled)

    st.subheader('Working With the Neural Network (Best Results)')
    # predictions = model.predict(arr)
    st.write("Predicted Values are:", predictions)
    # st.write("Real Values are:",y_test[:5])
    st.write("Aspect Ratios are:", predictions * 2)

    st.write(dc, ft, f3, Vcm, pdiss)


# dc = 0.0
# ft = 0.0
# f3 = 0.0
# Vcm = 0.0
# pdiss = 0.0

# SIDEBAR
st.sidebar.header('User Input Features')
dc = st.sidebar.number_input('DC Gain')
ft = st.sidebar.number_input('ft')
f3 = st.sidebar.number_input('f3')
Vcm = st.sidebar.number_input('Vcm')
pdiss = st.sidebar.number_input('PDiss', step=1e-6, format="%.7f")

st.sidebar.selectbox('Select a Model', ['Linear Regression Model', 'Gaussian Regression Model', 'SVR',
                                        'Decision Tree Regressor', 'KNN', 'Random Forest Regressor',
                                        'Neural Network (Best)'])


if st.sidebar.button('Calculate'):
    calculate(dc, ft, f3, Vcm, pdiss)

if st.sidebar.button('RESET'):
    pyautogui.hotkey("ctrl","F5")


# print(x)
# print(y)



# arr = np.array([[20.0438, 6080000.0000, 598810.0000, 1.6000, 0.0001]])
# ans = np.array([[0.0000, 0.0000, 0.0001, 0.0001, 0.0000]])




ans = np.array([[0.0000, 0.0000, 0.0001, 0.0001, 0.0000]])














# print("Predicted Values are:",predictions)
