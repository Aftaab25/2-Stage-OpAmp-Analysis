import pickle
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, \
    ExpSineSquared as Exp, DotProduct as Lin
from sklearn.metrics import mean_squared_error, mean_absolute_error


# df = pd.read_csv('2STAGEOPAMP_DATASET.csv')
#
# columns = ['Is4','Gm6','Gm4','Asp_1','Asp_2','Asp_3','Asp_4','Asp_5','Abs_Gain','Delay']
# df.drop(columns,axis='columns',inplace=True)
#
# # print(df.describe())
#
# # Split the Features and Labels and dividing into training and testing data
# labels = ['Wi1', 'Wi2', 'Wi3', 'Wi4', 'Wi5']
# features = ['DC Gain', 'ft', 'f3', 'Pdiss', 'Vcm']
# X = df.drop(labels, axis=1)
# y = df.drop(features, axis=1)
# # print(x)
# # print(y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)
#
#
# scaler = StandardScaler()
# scaler.fit(X_train)
#
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# y_train = y_train*(10**6)
# y_test = y_test*(10**6)
#
# load_gaussian_model = pickle.load(open('gaussian_model.pkl', 'rb'))
#
# y_pred_gp = load_gaussian_model.predict(X_test_scaled)
#
# mse_gp = mean_squared_error(y_test, y_pred_gp)
# mae_gp = mean_absolute_error(y_test, y_pred_gp)
# print("Mean Squared Error:", mse_gp)
# print("Root Mean Squared Error:", mse_gp ** 0.5)
# print("Mean Absolute Error:", mae_gp)
# from sklearn.metrics import r2_score
#
# print('R^2 Score :%.3f' % r2_score(y_test, y_pred_gp))

print(np.around([0.37, 1.64], decimals=1))
