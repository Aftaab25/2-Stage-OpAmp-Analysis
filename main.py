import pickle

import streamlit as st
# import pyautogui
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import keras

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, \
    ExpSineSquared as Exp, DotProduct as Lin
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(layout="wide")

# Header for the main Content
st.header('Aspect Ration Estimation of a Two-Stage Operational Amplifier')

# Reading the data
df = pd.read_csv('2STAGEOPAMP_DATASET.csv')
st.subheader('Description of the Dataset')
st.markdown("""
[Dataset used can be found here](https://github.com/Aftaab25/2-Stage-OpAmp-Analysis/blob/master/2STAGEOPAMP_DATASET.csv)
""")

columns = ['Is4', 'Gm6', 'Gm4', 'Asp_1', 'Asp_2', 'Asp_3', 'Asp_4', 'Asp_5', 'Abs_Gain', 'Delay']
df.drop(columns, axis='columns', inplace=True)

# Displaying the description of Dataset
st.write(df.describe())

# st.write(df)

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

y_train = y_train * (10 ** 6)
y_test = y_test * (10 ** 6)


# Neural Network
def neural_network():
    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)
    # st.write(test_user_scaled)
    # y_test = ans * (10 ** 6)
    model = load_model('model.h5')
    predictions = model.predict(test_user_scaled)

    st.subheader('Predictions from the Neural Network (Best Results)')
    # predictions = model.predict(arr)
    st.write("Predicted Values are:", predictions)
    # st.write("Real Values are:",y_test[:5])
    st.write("Aspect Ratios are:", predictions * 2)

    # st.write(dc, ft, f3, Vcm, pdiss)


# Linear Regression
def linear_regression():
    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)

    predictions = lr_model.predict(test_user_scaled)

    st.subheader('Prediction from Linear Regression Model')
    st.write("Predicted Values are:", predictions)
    st.write("Aspect Ratios are:", predictions * 2)


def gaussian_regression_model():
    load_gaussian_model = pickle.load(open('gaussian_model.pkl', 'rb'))

    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)

    predictions = load_gaussian_model.predict(test_user_scaled)

    st.subheader('Prediction from Gaussian Regression Model')
    st.write("Predicted Values are:", predictions)
    st.write("Aspect Ratios are:", predictions * 2)


def svr():
    svrregressor = SVR()
    mulregressor = MultiOutputRegressor(svrregressor)
    mulregressor.fit(X_train_scaled, y_train)

    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)

    predictions = mulregressor.predict(test_user_scaled)
    st.subheader('Prediction from SVR')
    st.write("Predicted Values are:", predictions)
    st.write("Aspect Ratios are:", predictions * 2)


def decision_tree_regressor():
    tree = DecisionTreeRegressor()
    tree.fit(X_train_scaled, y_train)

    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)

    predictions = tree.predict(test_user_scaled)
    st.subheader('Prediction from Decision Tree Regressor')
    st.write("Predicted Values are:", predictions)
    st.write("Aspect Ratios are:", predictions * 2)


def knn():
    knn = KNeighborsRegressor(n_neighbors=4)
    knn.fit(X_train_scaled, y_train)

    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)

    predictions = knn.predict(test_user_scaled)
    st.subheader('Prediction from KNN')
    st.write("Predicted Values are:", predictions)
    st.write("Aspect Ratios are:", predictions * 2)


def random_forest_regressor():
    model_RF = RandomForestRegressor(n_estimators=300, random_state=10)
    model_RF.fit(X_train_scaled, y_train)

    arr = np.array([[dc, ft, f3, Vcm, pdiss]])
    test_user_scaled = scaler.transform(arr)

    predictions = model_RF.predict(test_user_scaled)

    st.subheader('Prediction from Random Forest Regressor')
    st.write("Predicted Values are:", predictions)
    st.write("Aspect Ratios are:", predictions * 2)


# SIDEBAR
st.sidebar.header('User Input Features')
st.sidebar.caption('*Kindly make sure that the aspect ratios meet the requirements of the technology before designing.')
dc = st.sidebar.number_input('DC Gain')
ft = st.sidebar.number_input('Unity Gain Frequency (ft)')
f3 = st.sidebar.number_input('3-dB Frequency (f3)')
Vcm = st.sidebar.number_input('Common Mode Voltage (Vcm)')
pdiss = st.sidebar.number_input('Power Dissipation (PDiss)', step=1e-6, format="%.7f")

selected_model = st.sidebar.selectbox('Select a Model', ['Linear Regression Model', 'Gaussian Regression Model', 'SVR',
                                                         'Decision Tree Regressor', 'KNN', 'Random Forest Regressor',
                                                         'Neural Network (Best)'])


if st.sidebar.button('Calculate'):
    # calculate(dc, ft, f3, Vcm, pdiss)
    if selected_model == 'Neural Network (Best)':
        neural_network()
    elif selected_model == 'Linear Regression Model':
        linear_regression()
    elif selected_model == 'Gaussian Regression Model':
        gaussian_regression_model()
    elif selected_model == 'SVR':
        svr()
    elif selected_model == 'Decision Tree Regressor':
        decision_tree_regressor()
    elif selected_model == 'KNN':
        knn()
    elif selected_model == 'Random Forest Regressor':
        random_forest_regressor()

# if st.sidebar.button('RESET'):
#     pyautogui.hotkey("ctrl", "F5")

# print(x)
# print(y)


# arr = np.array([[20.0438, 6080000.0000, 598810.0000, 1.6000, 0.0001]])
# ans = np.array([[0.0000, 0.0000, 0.0001, 0.0001, 0.0000]])


ans = np.array([[0.0000, 0.0000, 0.0001, 0.0001, 0.0000]])

# print("Predicted Values are:",predictions)
