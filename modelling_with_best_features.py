'''

Create X_train and X_test using feature selection techniques (boruta and SelectKBest)

How to run: python3 modelling_with_best_features.py

'''
# ==================
# Import libraries
# ==================
import numpy as np
import pandas as pd
import pickle
import argparse

from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn import tree

from sklearn.feature_selection import SelectKBest, f_classif
from boruta import BorutaPy

import xgboost as xgb

# import warnings
# warnings.filterwarnings('ignore')

SEED = 42

def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def main():

    # =======================
    # LOAD DATA
    # =======================
    df = pd.read_csv('data/Parameters_90%stability.csv')
    df = df.drop(['Unnamed: 0'], axis = 1)
    X_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')
    X_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # ===============================
    # Drop cols with constant values
    # ===============================
    cols_to_drop = X_train.columns[X_train.nunique() <= 1].values
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    df = df.drop(columns=cols_to_drop)

    # =========================
    # Normalize
    # =========================
    X_train, X_test = normalize(X_train, X_test)

    # ============
    # SelectKBest
    # ============
    select_class = SelectKBest(f_classif, k=15)
    select_class.fit(X_train, y_train)
    X_train_kbest = select_class.transform(X_train)
    kbest_df = df.drop('Stability',axis=1).iloc[:,select_class.get_support()]
    kbest_df.columns.values
    selectkbest_features = kbest_df.columns.values

    # ============
    # Boruta
    # ============
    forest = RandomForestClassifier(random_state=SEED)
    feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=SEED)
    feat_selector.fit(X_train, y_train)

    # check selected features
    print(feat_selector.support_)
    # check ranking of features
    print(feat_selector.ranking_)

    X_train_boruta = feat_selector.transform(X_train)
    X_test_boruta = feat_selector.transform(X_test)
    print(X_train_boruta.shape, X_test_boruta)

    bor_feat = pd.DataFrame(feat_selector.support_, columns=['keep'])
    bor_feat['col'] = df.drop('Stability',axis=1).columns.values
    bor_feat['score'] = feat_selector.ranking_
    boruta_features = bor_feat[bor_feat['keep']==True]['col'].values
    print(boruta_features)


    # =============================
    # Save new datasets
    # =============================
    X_train = pd.read_csv('data/x_train.csv')
    X_test = pd.read_csv('data/x_test.csv')

    X_train_kbest = X_train[kbest_df.columns]
    X_test_kbest = X_test[kbest_df.columns]

    X_train_boruta = X_train[boruta_features]
    X_test_boruta = X_test[boruta_features]

    X_train_boruta.to_csv('data/x_train_boruta.csv', index=False)
    X_test_boruta.to_csv('data/x_test_boruta.csv', index=False)

    X_train_kbest.to_csv('data/x_train_kbest.csv', index=False)
    X_test_kbest.to_csv('data/x_test_kbest.csv', index=False)


if __name__ == '__main__':
    main()
