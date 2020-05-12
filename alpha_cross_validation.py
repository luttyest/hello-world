import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import time
import joblib
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn import model_selection

wine = pd.read_csv('data.csv', sep=',')

pd.isnull(wine).sum() > 0
#wine.replace([np.inf, -np.inf], np.nan, inplace=True)

wine.replace([np.inf, -np.inf], np.nan).dropna(how="all")

from sklearn.metrics import mean_squared_error
def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return mse
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return mse
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

X = wine.drop('tau1', axis=1).values
y = wine['tau1'].values

K = 10
#cross validation
alphas = [0.001, 0.01, 0.1, 1, 10]
kf = KFold(n_splits=K, shuffle=True, random_state=42)

for alpha in alphas:
    train_errors = []
    validation_errors = []
    sc = StandardScaler()
    X = sc.fit_transform(X)
    for train_index, val_index in kf.split(X, y):
        
        # split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # instantiate model
        model = MLPRegressor(activation='relu', solver='lbfgs',alpha=alpha, hidden_layer_sizes=(30,30,30), max_iter=1000)
        #calculate errors
        train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, model)
        
        # append to appropriate list
        train_errors.append(train_error)
        validation_errors.append(val_error)
    
    # generate report
    print('alpha: {:6} | mean(train_error): {:7} | mean(val_error): {}'.
          format(alpha,
                 round(np.mean(train_errors),10),
                 round(np.mean(validation_errors),10)))