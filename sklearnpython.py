#import packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import time

wine = pd.read_csv('dataComplex-20-10-10-16.CSV', sep=',')
pd.isnull(wine).sum() > 0
#wine.replace([np.inf, -np.inf], np.nan, inplace=True)

wine.replace([np.inf, -np.inf], np.nan).dropna(how="all")
X = wine.drop('tau1', axis=1)
y = wine['tau1']

#train and test splitting of data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


#apply scaler to opt
tic = time.perf_counter()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

mlpc = MLPRegressor(activation='relu', solver='lbfgs', alpha=0.001,
        hidden_layer_sizes=(100, 100, 100, 100, 100, 100, 100), max_iter=100000)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
#print(pred_mlpc)
X_temp = sc.inverse_transform(X_test)[:, 1]
print(X_temp)
print(r2_score(y_test, pred_mlpc))
#RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, pred_mlpc)))
plt.scatter(X_temp, pred_mlpc)
toc = time.perf_counter()
print(f"run in {toc - tic:0.4f} seconds")

from sklearn.externals import joblib

# Save to file in the current working directory
joblib_file = "joblib_model_total.pkl"
joblib.dump(mlpc, joblib_file)

#load
joblib_model = joblib.load(joblib_file)

# Calculate the accuracy and predictions
result = joblib_model.score(X_test, y_test)
print(result)

#predict saved model with new data
#pr = pd.read_csv("dataComplex-20-10-10-16.csv")
#print(pr)
#pred_cols = list(pr.columns.values)[:-1]

#print(pred_cols)
# apply the whole pipeline to data
#pred = pd.Series(mlpc.predict(pr[pred_cols]))
#print(pred)
