PATH_TO_ROOT = './'
import sys

sys.path.append(PATH_TO_ROOT)

# Base
import os
import pandas as pd
import numpy as np
import click
import math

from sklearn.metrics import mean_squared_error

# Execution
from src.data.storms_dataset import get_storms_df
import src.utils.pipeline as exe

# User parameters
DATA_DIR = './data'  # Path from project root
PREDICTIONS_PATH = 'predictions'
KFOLDS_FILE = "4Kfolds_202012220741.csv"


@click.command()
@click.option('--model_id')
@click.option('--debug', is_flag=True)
@click.option('--only_test', is_flag=True)
def main(model_id, debug, only_test):
    MODEL_ID = 'L1A_' + model_id if not debug else 'debug_L1A_' + model_id
    print('-'*50)
    print(f'- Concatenating L1 predictions for {MODEL_ID}')
    print('-' * 50)

    # Dataset
    dset_df = get_storms_df(os.path.join(PATH_TO_ROOT, DATA_DIR))
    dset_df.set_index('image_id', inplace=True, drop=True)
    dset_df = exe.load_kfolds(dset_df, os.path.join(PATH_TO_ROOT, DATA_DIR, KFOLDS_FILE))
    dset_df.set_index('image_id', inplace=True, drop=True)

    if not only_test:
        # Concatenate predictions
        files = [i for i in os.listdir(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH)) if i.startswith(MODEL_ID + "_fold_")]
        train_df = []
        for file in files:
            tmp_df = pd.read_csv(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, file), index_col=0)
            train_df.append(tmp_df.copy())
            # Evaluate
            y_pred = tmp_df.wind_speed.round().astype(int).values
            y_true = dset_df.loc[tmp_df.index, 'wind_speed'].values.astype(np.float32)
            print(f"File: {file} | RMSE: {math.sqrt(mean_squared_error(y_true, y_pred)):.4}")
        train_df = pd.concat(train_df, axis=0)
        train_df = train_df.loc[dset_df.index[dset_df.train]]

        # Evaluate
        y_pred = train_df.wind_speed.round().astype(int).values
        y_true = dset_df.loc[train_df.index, 'wind_speed'].values.astype(np.float32)
        print(f"Model: {MODEL_ID} | RMSE: {math.sqrt(mean_squared_error(y_true, y_pred)):.4}")

        # Save folds predictions
        filepath = os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, MODEL_ID + f"_folds.csv.gz")
        print(f"Saving FOLDS predictions: {filepath}")
        train_df.to_csv(filepath, index=True)

    # Concatenate predictions
    files = [i for i in os.listdir(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH)) if i.startswith(MODEL_ID + "_test")]
    test_df = []
    for file in files:
        tmp_df = pd.read_csv(os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, file), index_col=0)
        test_df.append(tmp_df.copy())
    test_df = pd.concat(test_df, axis=0)
    test_df = test_df.groupby('image_id').mean()

    # Save test predictions
    filepath = os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, MODEL_ID + f"_test.csv.gz")
    print(f"Saving TEST predictions: {filepath}")
    test_df.to_csv(filepath, index=True)

    # Save submission
    filepath = os.path.join(PATH_TO_ROOT, PREDICTIONS_PATH, MODEL_ID + f"_submission.csv")
    sub_df = test_df.copy()
    sub_df = sub_df.round().astype(int)
    print(f"Saving SUBMISSION predictions: {filepath}")
    sub_df.to_csv(filepath, index=True)

    print('')


if __name__ == '__main__':
    main()
