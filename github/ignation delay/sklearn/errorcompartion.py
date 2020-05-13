#import packages
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import time


tic = time.perf_counter()
wine = pd.read_csv('dataComplextestC1.csv', sep=',')

X = wine.drop('tau1', axis=1)
y = wine['tau1']

#create error compration csv file
train_R = []
train_MSE = []
test_R = []
test_MSE = []

hiddenlayerinit = (30, 30)

i = 0
hiddenlayerwanted = 6
while i < hiddenlayerwanted:
    # hidden layer modify

   # train and test splitting of data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    #apply scaler to opt
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    mlpc = MLPRegressor(activation='relu', solver='lbfgs', alpha=0.001,
                        hidden_layer_sizes=hiddenlayerinit, max_iter=1000000000000000)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)
    #print(pred_mlpc)
    #X_temp = sc.inverse_transform(X_test)[:,0]
    pred_train = mlpc.predict(X_train)
    #train set
    train_R.append(r2_score(y_train, pred_train))
    #RMSE
    train_MSE.append(np.sqrt(metrics.mean_squared_error(y_train, pred_train)))

    # test set
    test_R.append(r2_score(y_test, pred_mlpc))
    #RMSE
    test_MSE.append(np.sqrt(metrics.mean_squared_error(y_test, pred_mlpc)))

    hiddenlayerinit = list(hiddenlayerinit)
    hiddenlayerinit.append(30)
    hiddenlayerinit = tuple(hiddenlayerinit)
    i += 1


with open("errorcompare.csv", 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["train_R", "train_RSE", "test_R", "test_RSE"])
    rows = zip(train_R, train_MSE, test_R, test_MSE)
    writer.writerows(rows)

toc = time.perf_counter()

print(f"task took {toc - tic:0.4f} seconds")

#plt.scatter(X_temp,pred_mlpc)

# print("weights between input and first hidden layer:")
# print(mlpc.coefs_[0])
# print("\nweights between first hidden and second hidden layer:")
# print(mlpc.coefs_[1])
