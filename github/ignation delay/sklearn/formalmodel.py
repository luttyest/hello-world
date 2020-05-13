#import packages
from sklearn.externals import joblib
import pandas as pd
import seaborn as sns
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

# k fold cross validation

tic = time.perf_counter()

# different alpha levels
alphas = [0.001, 0.01, 0.1, 1, 10]
print('All errors are RMSE')

wine = pd.read_csv('dataComplextestC1.csv', sep=',')
pd.isnull(wine).sum() > 0
#wine.replace([np.inf, -np.inf], np.nan, inplace=True)

wine.replace([np.inf, -np.inf], np.nan).dropna(how="all")
data = wine.drop('tau1', axis=1)
target = wine['tau1']

K = 10
#cross validation
kf = KFold(n_splits=K, shuffle=True, random_state=42)

for alpha in alphas:
    train_errors = []
    validation_errors = []
    for train_index, val_index in kf.split(data, target):
        
        # split data
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = target[train_index], target[val_index]

        # instantiate model
        model = MLPRegressor(activation='relu', solver='lbfgs',alpha=alpha, hidden_layer_sizes=(30,30,30,30,30), max_iter=10000000000000000000000)
        
        #calculate errors
        train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, model)
        
        # append to appropriate list
        train_errors.append(train_error)
        validation_errors.append(val_error)
    
    # generate report
    print('alpha: {:6} | mean(train_error): {:7} | mean(val_error): {}'.
          format(alpha,
                 round(np.mean(train_errors),4),
                 round(np.mean(validation_errors),4)))

    
# kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.20, random_state=100)
# model_shufflecv = MLPRegressor(activation='relu', solver='lbfgs',alpha=0.001, hidden_layer_sizes=(30,30,30,30,30), max_iter=10000000000000000000000)
# sc = StandardScaler()
# X = sc.fit_transform(X)
# results_4 = model_selection.cross_val_score(model_shufflecv, X, y, cv=kfold2)
# print("Accuracy: %.2f%% (%.2f%%)" % (results_4.mean()*100.0, results_4.std()*100.0))





