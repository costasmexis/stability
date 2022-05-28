'''

Export 'rules' from decision tree classifier

'''

import numpy as np
import pandas as pd
import pickle
import argparse

from matplotlib import pyplot as plt

from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import tree
from sklearn.tree import _tree

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

SEED = 42

def main():
    # ===================
    # Read data files
    # ===================
    df = pd.read_csv('data/Parameters_90%stability.csv')
    df = df.drop(['Unnamed: 0'], axis = 1)

    X_train = pd.read_csv('data/x_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    X_test = pd.read_csv('data/x_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    def normalize(X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    # ===================
    # Normalize data
    # ===================
    X_train, X_test = normalize(X_train, X_test)

    parser = argparse.ArgumentParser(description='Args model from which rules will be extracted')
    parser.add_argument('-model','--model', help='ex. svc_tree_model.sav', required=True)
    args = parser.parse_args()

    # ===================
    # Load model from disk
    # ===================
    filename = "models/"+args.model
    model = pickle.load(open(filename, 'rb'))
    print(model)

    def get_rules(tree, feature_names, class_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if class_names is None:
                rule += "response: "+str(np.round(path[-1][0][0][0],3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                # rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
                rule += f"class: {class_names[l]}"
            rule += f" | based on {path[-1][1]:,} samples"
            rules += [rule]

        return rules

    feature_names = df.drop('Stability',axis=1).columns.values

    rules = get_rules(model, feature_names=feature_names, class_names=['False','True'])

    textfile = open("rules/"+args.model+"_all_rules.txt", "w")
    for element in rules:
        textfile.write(element + "\n")
    textfile.close()

    # rules as DataFrame for easy handling
    rules = pd.DataFrame(rules)
    rules.rename(columns = {0:'rule'}, inplace = True)
    rules['rule'] = rules['rule'].astype('string')
    rules['index_of_true'] = rules['rule'].str.find('True')
    rules['class'] = rules['index_of_true'] != -1

    rules[rules['class']==True]['rule'].to_csv("rules/"+args.model+'_stability_rules.txt', header=None, index=None, sep=' ', mode='a')

if __name__ == '__main__':
    main()