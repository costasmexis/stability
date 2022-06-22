# ==================
# Import libraries
# ==================
import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn import tree

import xgboost as xgb

SEED = 42

value = input("Please enter 1(simple), 2(boruta), 3(kbest) dataset:\n")
print(f'You entered {value}')

if(value=="1"):
    value="simple"
    df = pd.read_csv('data/Parameters_90%stability.csv')
    df = df.drop(['Unnamed: 0'], axis = 1)

    X_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    X_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
elif(value=='2'):
    value="boruta"
    X_train = pd.read_csv('data/x_train_boruta.csv')
    y_train = pd.read_csv('data/y_train.csv')

    X_test = pd.read_csv('data/x_test_boruta.csv')
    y_test = pd.read_csv('data/y_test.csv')
elif(value=='3'):
    value="kbest"
    X_train = pd.read_csv('data/x_train_kbest.csv')
    y_train = pd.read_csv('data/y_train.csv')

    X_test = pd.read_csv('data/x_test_kbest.csv')
    y_test = pd.read_csv('data/y_test.csv')
else:
    print("Wrong choice. Try again!")

print("Train: ", X_train.shape, "Test: ", X_test.shape)

def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def print_scores(y_true, y_pred):
    print('ROCAUC score:',roc_auc_score(y_true, y_pred))
    print('Accuracy score:',accuracy_score(y_true, y_pred))
    print('F1 score:',f1_score(y_true, y_pred))
    print('Precision score:',precision_score(y_true, y_pred))
    print('Recall:',recall_score(y_true, y_pred))
    print("\n\n")

def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(model)
    print_scores(y_test, y_pred)
    return score, y_pred

def tune_model(model, param_grid, n_iter, X_train, y_train):
    grid = RandomizedSearchCV(model, param_grid, verbose=20,
        scoring='roc_auc', cv=3, n_iter=n_iter)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model

feature_names = X_train.columns
X_train, X_test = normalize(X_train, X_test)

# ============================
# SMOTE
# ============================
from imblearn.over_sampling import SMOTE

over = SMOTE(random_state=SEED)
X_train_res, y_train_res = over.fit_resample(X_train, y_train.values.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))


X_train_res = pd.DataFrame(data=X_train_res, columns=feature_names)
y_train_res = pd.DataFrame(data=y_train_res)
y_train_res.rename(columns = {0:'Stability'}, inplace = True)

X_train_res.to_csv('data/x_train_smote.csv', index=False)
y_train_res.to_csv('data/y_train_smote.csv', index=False)

