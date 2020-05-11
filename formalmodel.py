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
from sklearn import model_selection

tic = time.perf_counter()

wine = pd.read_csv('dataComplextestC1.csv', sep=',')
pd.isnull(wine).sum() > 0
#wine.replace([np.inf, -np.inf], np.nan, inplace=True)

wine.replace([np.inf, -np.inf], np.nan).dropna(how="all")
X = wine.drop('tau1', axis=1)
y = wine['tau1']

#cross validation
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression


kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.20, random_state=100)
model_shufflecv = MLPRegressor(activation='relu', solver='lbfgs',alpha=0.001, hidden_layer_sizes=(30,30,30,30,30), max_iter=10000000000000000000000)
sc = StandardScaler()
X = sc.fit_transform(X)
results_4 = model_selection.cross_val_score(model_shufflecv, X, y, cv=kfold2)
print("Accuracy: %.2f%% (%.2f%%)" % (results_4.mean()*100.0, results_4.std()*100.0))





