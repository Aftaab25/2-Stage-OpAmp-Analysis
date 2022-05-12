import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

lr_model = pickle.load(open('lr_model.pkl', 'rb'))
predictions = lr_model.predict(X_test_scaled[:5])
y_pred_lr = lr_model.predict(X_test_scaled)
print("Predicted Values are:",predictions)