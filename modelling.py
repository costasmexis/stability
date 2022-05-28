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

import xgboost as xgb

SEED = 42

df = pd.read_csv('data/Parameters_90%stability.csv')
df = df.drop(['Unnamed: 0'], axis = 1)

X_train = pd.read_csv('data/x_train.csv')
y_train = pd.read_csv('data/y_train.csv')

X_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')

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
    grid = RandomizedSearchCV(model, param_grid, verbose=2,
        scoring='roc_auc', cv=5, n_iter=n_iter)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model

X_train, X_test = normalize(X_train, X_test)

def svc():

    # =================
    # SVC
    # =================

    param_grid_svc = {'C': [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 1, 1.5, 2, 2.5, 3, 5, 10, 12, 20, 25, 50],
                'gamma': [0.002, 0.003, 0.004, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
                'kernel': ['rbf', 'linear']
    }

    svc = SVC(random_state=SEED)
    best_svc = tune_model(svc, param_grid_svc, 1000, X_train, y_train.values.ravel())
    score, y_pred = run_model(best_svc, X_train, y_train.values.ravel(),
        X_test, y_test.values.ravel())

    # save the model to disk
    filename = 'svc_model.sav'
    pickle.dump(best_svc, open(filename, 'wb'))

def tree():

    # =================
    # DecisionTree
    # =================

    param_grid_tree = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 40, 50, 100],
        'min_samples_leaf': [5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100],
        'criterion': ["gini", "entropy"]
    }

    dec_tree = DecisionTreeClassifier(random_state=SEED)
    best_tree = tune_model(dec_tree, param_grid_tree, 1000, X_train, y_train.values.ravel())
    score, y_pred = run_model(best_tree, X_train, y_train.values.ravel(),
        X_test, y_test.values.ravel())

    filename = 'tree_model.sav'
    pickle.dump(best_tree, open(filename, 'wb'))
    
    plt.figure(figsize=(30,30))  # set plot size (denoted in inches)
    tree.plot_tree(model, filled=True, class_names=['0','1'])
    plt.savefig('dcs_tree.png')


def catboost():

    # ===================
    # CatBoost
    # ===================
    cat = CatBoostClassifier()
    cat.fit(X_train, y_train.values.ravel())
    score, y_pred = run_model(cat, X_train, y_train.values.ravel(),
        X_test, y_test.values.ravel())

    filename = 'catboost_model.sav'
    pickle.dump(cat, open(filename, 'wb'))
    

def xgboost():

    # ==========================
    # XGBClassifier
    # ==========================

    param_grid_xgb = {
        'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
        'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight' : [ 1, 3, 5, 7 ],
        'gamma': [ 0.0, 0.01, 0.05, 0.1, 0.2 , 0.3, 0.4 ],
        'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ],
        'n_estimators' : [10, 25, 50, 100, 150, 200, 300, 500]
    }
    
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, 
        eval_metric='logloss', random_state=SEED)

    best_xgb = tune_model(xgb_model, param_grid_xgb, 100, X_train, y_train.values.ravel())
    score, y_pred = run_model(best_xgb, X_train, y_train.values.ravel(),
        X_test, y_test.values.ravel())

    filename = 'xgb_model.sav'
    pickle.dump(best_xgb, open("models/"+filename, 'wb'))
    
