PATH_TO_ROOT = './'
import sys

sys.path.append(PATH_TO_ROOT)

import numpy as np
import pandas as pd
import lightgbm as lgb
import click
from tqdm import tqdm
import math
import hashlib
import json
import os

from src.data.storms_dataset import get_storms_df

import math
from sklearn.metrics import mean_squared_error
rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))

# List of models with 1 level predictions
MODEL_VERSION = 'FINAL'

MODELS_LIST = ['L1A_A2', 'L1A_B1', 'L1A_C1',
               'L1A_B2', 'L1A_A1', 'L1A_C3', 'L1A_B3',
               'L1A_A3', 'L1A_C2', 'L1A_D1']
MAIN_PRED_LIST = ['L1A_A6', 'L1A_A4', 'L1A_A5', 'L1A_A7', ]

TRAIN_FEATURES_TEMPLATE = '{predictions_folder}/{model_name}_folds.csv.gz'
TEST_FEATURES_TEMPLATE = '{predictions_folder}/{model_name}_test.csv.gz'

MODELS_PATH = 'models'


def load_data(data_folder, models_list, avg, predictions_folder='predictions'):

    print('Loading 1st level model features...')
    train_features_df = pd.read_csv(TRAIN_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=models_list[0]), index_col=0)
    for model_name in models_list[1:]:
        if avg:
            train_features_df = train_features_df + pd.read_csv(TRAIN_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=model_name), index_col=0)
        else:
            train_features_df = pd.concat((train_features_df, pd.read_csv(TRAIN_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=model_name), index_col=0)), axis=1)
    print('Train data shape - ', train_features_df.shape)
    if avg:
        train_features_df = train_features_df / len(models_list)
    else:
        train_features_df.columns = [s1+'_wind_speed' for s1 in models_list]

    print(train_features_df.head())

    test_features_df = pd.read_csv(TEST_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=models_list[0]), index_col=0)
    for model_name in models_list[1:]:
        if avg:
            test_features_df = test_features_df + pd.read_csv(TEST_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=model_name), index_col=0)
        else:
            test_features_df = pd.concat((test_features_df, pd.read_csv(TEST_FEATURES_TEMPLATE.format(predictions_folder=predictions_folder, model_name=model_name),index_col=0)), axis=1)
    print('Test data shape - ', test_features_df.shape)
    if avg:
        test_features_df = test_features_df / len(models_list)
    else:
        test_features_df.columns = [s1+'_wind_speed' for s1 in models_list]
    print(test_features_df.head())

    return train_features_df, test_features_df


def get_prev_data_df(data_folder, dset_df, prev_data_parameters):

    # Hash parameters to get filename
    dhash = hashlib.md5()
    encoded = json.dumps(prev_data_parameters, sort_keys=True).encode()
    dhash.update(encoded)
    prev_data_filepath = os.path.join(data_folder, f'prev_data_{dhash.hexdigest()}.csv.gz')

    # Check if file already exists
    if os.path.isfile(prev_data_filepath):
        return pd.read_csv(prev_data_filepath)

    # prepare dataframes
    df = dset_df.copy()
    if 'image_id' in df.columns:
        df.set_index('image_id', inplace=True, drop=True)
    # Create storms dataset
    storm_df = df[['storm_id', 'storm_duration']].copy()
    storm_df['image_id'] = storm_df.index
    storm_df.set_index(['storm_id', 'storm_duration'], inplace=True, drop=True)

    # Get previous data indexes
    gaps = prev_data_parameters['gaps']
    margin = prev_data_parameters['margin']
    result_df = []
    result_df.append(df.index.values)
    for gap in gaps:
        new_rows = []
        for iix in tqdm(df.index):
            try:
                new_rows.append(storm_df.loc[(df.loc[iix].storm_id,
                                              df.loc[iix].storm_duration - gap), 'image_id'][0])
            except KeyError:
                try:
                    new_rows.append(
                        storm_df.loc[(df.loc[iix].storm_id,
                                      df.loc[iix].storm_duration - gap - margin), 'image_id'][0])
                except KeyError:
                    try:
                        new_rows.append(storm_df.loc[(df.loc[iix].storm_id,
                                                      df.loc[iix].storm_duration - gap + margin), 'image_id'][0])
                    except KeyError:
                        new_rows.append('unk')
        result_df.append(new_rows)

    result_df = pd.DataFrame(np.array(result_df).T, columns=['image_id', ] + ['_' + str(s1) for s1 in gaps])

    # Save
    result_df.to_csv(prev_data_filepath, index=False)

    return result_df


def add_prev_data(prev_data_df, train_features_df, test_features_df, columns=None, verbose=False):
    if columns is None:
        columns = train_features_df.columns

    # Concatenate predictions and add unk
    all_l1_preds = pd.concat([train_features_df, test_features_df], axis=0)
    unk_row = all_l1_preds.iloc[0] * 0
    unk_row.name = 'unk'
    all_l1_preds = all_l1_preds.append(unk_row)

    new_train_data = [train_features_df, ]
    new_test_data = [test_features_df, ]
    if verbose:
        ite = tqdm(columns)
    else:
        ite = columns
    for column in ite:
        new_train_data.append(pd.DataFrame({
            str(column) + str(s1): all_l1_preds.loc[
                prev_data_df.loc[train_features_df.index, s1].values, column].values
            for s1 in prev_data_df.columns}, index=train_features_df.index))

        new_test_data.append(pd.DataFrame({
            str(column) + str(s1): all_l1_preds.loc[
                prev_data_df.loc[test_features_df.index, s1].values, column].values
            for s1 in prev_data_df.columns}, index=test_features_df.index))
    train_features_df = pd.concat(new_train_data, axis=1)
    test_features_df = pd.concat(new_test_data, axis=1)
    return train_features_df, test_features_df


@click.command()
@click.option('--data_folder', default='data')
@click.option('--avg', is_flag=True)
@click.option('--num_iterations', default=10000)
@click.option('--predictions_folder', default='predictions')
def main(data_folder, avg, num_iterations, predictions_folder):

    data_folder, avg, num_iterations, predictions_folder = 'data', False, 10000, 'predictions'

    MODEL_TYPE = 'L2A'
    TRAINING_VERSION = ''
    OUTPUT_NAME = f"{MODEL_TYPE}_{MODEL_VERSION}{TRAINING_VERSION}"
    OUTPUT_PATH = predictions_folder

    models_list = MODELS_LIST.copy()
    print('Models: ', models_list)
    print('MAIN Models: ', MAIN_PRED_LIST)

    # Load L1 predictions
    train_L1_pred_df, test_L1_pred_df = load_data(data_folder, models_list,
                                                  avg=avg, predictions_folder=predictions_folder)

    # Load MAIN predictions. These model's predictions would be average
    if len(MAIN_PRED_LIST) > 0:
        train_MAIN_pred_df, test_MAIN_pred_df = load_data(data_folder, MAIN_PRED_LIST,
                                                          avg=True, predictions_folder=predictions_folder)
        train_MAIN_pred_df.columns = ['MAIN_'+str(s1) for s1 in train_MAIN_pred_df.columns]
        test_MAIN_pred_df.columns = ['MAIN_' + str(s1) for s1 in test_MAIN_pred_df.columns]

    # Load main dataset
    dset_df = get_storms_df(data_folder)
    dset_df.set_index('image_id', inplace=True, drop=True)
    # Create storms dataset
    storm_df = dset_df[['storm_id', 'storm_duration']].copy()
    storm_df['image_id'] = storm_df.index
    storm_df.set_index(['storm_id', 'storm_duration'], inplace=True, drop=True)

    # Read prev_data_df: Dataframe with image_id for previous data
    prev_data_parameters = {
        'gaps': [1, 2, 4, 6, 8, 10, 12, 16, 20, 24],  # Previous data gap, in hours
        'margin': 0.5  # If previous data doesn't exist, search for +/- margin, in hours
    }
    prev_data_df = get_prev_data_df(data_folder, dset_df, prev_data_parameters)
    prev_data_df.set_index('image_id', inplace=True, drop=True)

    # Load FOLDS
    folds = pd.read_csv(f"{data_folder}/4Kfolds_202012220741.csv")
    folds.set_index('image_id', drop=True, inplace=True)
    folds = folds.loc[train_L1_pred_df.index]
    folds.sort_index(inplace=True)

    ## Build features dataframe
    # Add previous data
    train_features_df, test_features_df = \
        add_prev_data(prev_data_df, train_L1_pred_df, test_L1_pred_df, verbose=True)
    if len(MAIN_PRED_LIST) > 0:
        # Get MAIN features
        train_main_feat_df, test_main_feat_df = add_prev_data(prev_data_df, train_MAIN_pred_df, test_MAIN_pred_df)
        # Add MAIN for train set
        new_train_data = [train_features_df, train_main_feat_df]
        train_features_df = pd.concat(new_train_data, axis=1)
        # Add MAIN for test set
        new_test_data = [test_features_df, test_main_feat_df]
        test_features_df = pd.concat(new_test_data, axis=1)

    train_features_df.sort_index(inplace=True)
    test_features_df.sort_index(inplace=True)
    # Add extra features
    def add_extra_features(features_df):
        features_df['ocean'] = dset_df.loc[features_df.index, 'ocean']
        features_df['storm_duration'] = dset_df.loc[features_df.index, 'storm_duration']
        return features_df
    train_features_df = add_extra_features(train_features_df)
    test_features_df = add_extra_features(test_features_df)

    # Final columns
    feature_columns = [s1 for s1 in train_features_df.columns if s1 not in []]

    # Train the model
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'rmse',
        'metric': {'rmse'},
        'num_threads': 8,
        'num_iterations': num_iterations,
        # Core parameters
        'learning_rate': 0.005,
        # Learning control parameters
        'max_depth': 5,
        'feature_fraction': 0.8,  # colsample_bytree
        'bagging_freq': 1,
        'bagging_fraction': 0.6,  # subsample
        'num_leaves': 16,
        'min_data_in_leaf': 15,
        'verbosity': -1,
    }

    all_valid_y = []
    all_valid_field_ids = []
    all_predicts = []
    all_valid_predicts = []
    best_iterations = []
    fe = []
    for fold in sorted(folds.fold.unique()):
        # print(f'Fold - {fold}')
        train_field_ids_i = folds[folds.fold != fold].index.values
        val_field_ids_i = folds[(folds.fold == fold) & folds.for_validation].index.values

        train_data = train_features_df[feature_columns].astype(np.float32)

        X_train, X_valid = train_data.loc[train_field_ids_i, :].values, \
                           train_data.loc[val_field_ids_i, :].values
        y_train, y_valid = dset_df.loc[train_field_ids_i, 'wind_speed'].values, \
                           dset_df.loc[val_field_ids_i, 'wind_speed'].values,

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        model_lgb = lgb.train(params, train_data, valid_sets=[valid_data],
                              early_stopping_rounds=50, verbose_eval=False)
        best_iterations.append(model_lgb.best_iteration)
        print(f"Fold - {fold} - {model_lgb.best_score['valid_0']['rmse']:.4}@{model_lgb.best_iteration}")

        # Save model
        #filepath = os.path.join(MODELS_PATH, OUTPUT_NAME + f"_model_{fold}.gbm")
        #model_lgb.save_model(filepath, model_lgb.best_iteration)

        # As a little trick, at prediction time, substitute predicted past values for actual values.

        # Update validation L1 with actual known wind_speed values
        VALID_train_L1_pred_df = train_L1_pred_df.copy()
        for column in VALID_train_L1_pred_df.columns:
            VALID_train_L1_pred_df.loc[train_field_ids_i, column] = dset_df.loc[
                train_field_ids_i, 'wind_speed']

        # Update validation MAIN with actual known wind_speed values
        if len(MAIN_PRED_LIST) > 0:
            VALID_train_MAIN_pred_df = train_MAIN_pred_df.copy()
            VALID_train_MAIN_pred_df.loc[train_field_ids_i, 'MAIN_wind_speed'] = dset_df.loc[
                train_field_ids_i, 'wind_speed']

        ## Build features dataframe
        # Add previous data
        VALID_train_features_df, INFER_test_features_df = \
            add_prev_data(prev_data_df, VALID_train_L1_pred_df, test_L1_pred_df, verbose=True)

        # Get MAIN features
        if len(MAIN_PRED_LIST) > 0:
            train_main_feat_df, test_main_feat_df = add_prev_data(prev_data_df,
                                                                  VALID_train_MAIN_pred_df, test_MAIN_pred_df)
        # Add MAIN for train set
        if len(MAIN_PRED_LIST) > 0:
            new_train_data = [VALID_train_features_df, train_main_feat_df]
            VALID_train_features_df = pd.concat(new_train_data, axis=1)
        else:
            VALID_train_features_df = VALID_train_features_df
        VALID_train_features_df.sort_index(inplace=True)

        # Add MAIN for test set
        if len(MAIN_PRED_LIST) > 0:
            new_test_data = [INFER_test_features_df, test_main_feat_df]
            INFER_test_features_df = pd.concat(new_test_data, axis=1)
        INFER_test_features_df.sort_index(inplace=True)

        # Add extra features
        VALID_train_features_df = add_extra_features(VALID_train_features_df)
        INFER_test_features_df = add_extra_features(INFER_test_features_df)

        # Select columns
        VALID_train_features_df = VALID_train_features_df[feature_columns].astype(np.float32)
        INFER_test_features_df = INFER_test_features_df[feature_columns].astype(np.float32)

        # Make predictions for VALID data set
        y_valid_pred = model_lgb.predict(VALID_train_features_df.loc[val_field_ids_i],
                                         num_iteration=model_lgb.best_iteration)

        # Make predictions for TEST data set
        ypred = model_lgb.predict(INFER_test_features_df, num_iteration=model_lgb.best_iteration)

        all_valid_field_ids.extend(val_field_ids_i)
        all_predicts.append(ypred)
        all_valid_y.extend(y_valid)
        all_valid_predicts.append(y_valid_pred)

        fe.append(model_lgb.feature_importance())
    all_valid_predicts = np.concatenate(all_valid_predicts)
    all_valid_y = np.array(all_valid_y)
    print(f'Validation RMSE - {rmse(all_valid_y, all_valid_predicts):.4f}')

    # Save folds predictions
    valid_preds_df = pd.DataFrame(np.vstack(all_valid_predicts), index=all_valid_field_ids, columns=['wind_speed'])
    valid_preds_df.index.name = 'image_id'
    filepath = os.path.join(OUTPUT_PATH, OUTPUT_NAME + '_folds.csv.gz')
    print(f"Saving VALIDATION predictions: {filepath}")
    valid_preds_df.to_csv(filepath, index=True)

    # Save test predictions
    result = np.zeros_like(all_predicts[0])
    for predict in all_predicts:
        result += predict
    result = result / len(all_predicts)
    test_preds_df = pd.DataFrame(result, index=test_features_df.index, columns = ['wind_speed'])
    test_preds_df.index.name = 'image_id'
    filepath = os.path.join(OUTPUT_PATH, OUTPUT_NAME + '_test.csv.gz')
    print(f"Saving TEST predictions: {filepath}")
    test_preds_df.to_csv(filepath, index=True)

    # Save submission
    filepath = os.path.join(OUTPUT_PATH, OUTPUT_NAME + f"_submission.csv")
    sub_df = test_preds_df.copy()
    sub_df = sub_df.round().astype(int)
    print(f"Saving SUBMISSION predictions: {filepath}")
    sub_df.to_csv(filepath, index=True)


if __name__ == '__main__':
    main()
