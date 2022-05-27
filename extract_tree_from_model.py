'''

In this script a DecisionTreeClassifier is trained on X_test and y_pred (produced by another classifier)
in order to extract the rules learned by the (another) classifier.

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

import xgboost as xgb

SEED = 42

def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def tune_model(model, param_grid, n_iter, X_train, y_train):
    grid = RandomizedSearchCV(model, param_grid, verbose=2,
        scoring='roc_auc', cv=5, n_iter=n_iter)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model

def print_scores(y_true, y_pred):
    print('ROCAUC score:',roc_auc_score(y_true, y_pred))
    print('Accuracy score:',accuracy_score(y_true, y_pred))
    print('F1 score:',f1_score(y_true, y_pred))
    print('Precision score:',precision_score(y_true, y_pred))
    print('Recall:',recall_score(y_true, y_pred))
    print("\n\n")


# ===================
# Read data files
# ===================
df = pd.read_csv('data/Parameters_90%stability.csv')
df = df.drop(['Unnamed: 0'], axis = 1)

X_train = pd.read_csv('data/x_train.csv')
y_train = pd.read_csv('data/y_train.csv')

X_test = pd.read_csv('data/x_test.csv')
y_test = pd.read_csv('data/y_test.csv')

print("Train: ", X_train.shape, "Test: ", X_test.shape)

# ===================
# Normalize data
# ===================
X_train, X_test = normalize(X_train, X_test)

parser = argparse.ArgumentParser(description='Args model from which rules will be extracted')
parser.add_argument('-model','--model', help='Model', required=True)
# parser.add_argument('-file_name','--file_name', help='ex. svc_tree_model.sav', required=True)
# parser.add_argument('-figure_name','--figure_name', help='ex. svc_tree_graph.png', required=True)
args = parser.parse_args()

# ===================
# Load model from disk
# ===================
filename = "models/"+args.model
model = pickle.load(open(filename, 'rb'))
print(model)

# Make predictions
y_pred = model.predict(X_test)
print_scores(y_test, y_pred)

def train_tree(file_name, figure_name, tuning=True):

    # =================
    # DecisionTree
    # =================

    if(tuning==True):
        param_grid_tree = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 40, 50, 100],
            'min_samples_leaf': [5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100],
            'criterion': ["gini", "entropy"]
        }
        dec_tree = DecisionTreeClassifier(random_state=SEED)
        best_tree = tune_model(dec_tree, param_grid_tree, 1000, X_test, y_pred)
    else:
        best_tree = DecisionTreeClassifier(random_state=SEED)
        best_tree.fit(X_test, y_pred)

    filename = file_name
    pickle.dump(best_tree, open(filename, 'wb'))
    
    plt.figure(figsize=(30,30))  # set plot size (denoted in inches)
    tree.plot_tree(best_tree, filled=True, class_names=['0','1'])
    plt.savefig(figure_name)

# =======================================================
# Train a DecisionTreeClassifier using X_test and y_pred
# =======================================================
train_tree("models/"+args.model+"_tree_model.sav", "figures/"+args.model+"_tree_graph.png", False)


