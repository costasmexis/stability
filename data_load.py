'''
This script loads the whole dataset and exports the TRAIN / SUB-TRAIN / VALIDATION / TEST sets

HOW TO RUN?

python3 data_load.py -input "PATH/TO/INPUT_DATASET.csv"
ex. python3 data_load.py -input "data/Parameters_90%stability.csv"

'''

# =============================================================================
# Import Libraries                 
# =============================================================================
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
# =============================================================================
# Define functions
# =============================================================================
def read_csv(file_name):
    return pd.read_csv(file_name)

# =============================================================================
# Load data files
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Args to insert')
    parser.add_argument('-input','--input', help='Dataset', required=True)
    args = parser.parse_args()

    df = read_csv(args.input)
    df = df.drop(['Unnamed: 0'], axis = 1)

    # Load X and Y 
    X = df.drop(['Stability'], axis = 1)
    y = df['Stability']

    # =============================================================================
    #                   Split to TRAIN, VALIDATION and TEST sets
    # =============================================================================

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        stratify=y, random_state=42)

    x_train.to_csv('data/x_train.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)

    x_test.to_csv('data/x_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)


    x_sub_train, x_val, y_sub_train, y_val = train_test_split(x_train, y_train, test_size=0.20,
                                                        stratify=y_train, random_state=42)

    x_sub_train.to_csv('data/x_sub_train.csv', index=False)
    y_sub_train.to_csv('data/y_sub_train.csv', index=False)

    x_val.to_csv('data/x_val.csv', index=False)
    y_val.to_csv('data/y_val.csv', index=False)

if __name__ == '__main__':
    main()