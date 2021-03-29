PATH_TO_ROOT = './'
import sys

sys.path.append(PATH_TO_ROOT)

# Base
import os
import pandas as pd
import subprocess
import click

# Execution
import src.utils.pipeline as exe
from src.utils.logging import init_logger

# Specific
from src.data.storms_dataset import get_storms_df

import math
from sklearn.metrics import mean_squared_error
rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))


# User parameters
MODEL_MODULE_PATH = "src/models"
KFOLDS_FILE = "4Kfolds_202012220741.csv"
OUTPUTS_PATH = 'predictions'
MODELS_PATH = 'models'


@click.command()
@click.option('--model_id')
@click.option('--folds', default=None)
@click.option('--debug', is_flag=True)
def main(model_id, folds, debug):
    MODEL_ID = model_id

    # Script parameters
    MODEL_NAME = f"model_L1A_{MODEL_ID}"
    DATA_DIR = "data"
    PATH_TO_LOG = ".log"

    print('*' * 80)
    prefix='debug_' if debug else ''
    # Init logger
    if folds is None:
        log_filename = os.path.join(PATH_TO_ROOT, PATH_TO_LOG, prefix + MODEL_NAME + '.log')
    else:
        log_filename = os.path.join(PATH_TO_ROOT, PATH_TO_LOG, prefix + MODEL_NAME + f'_{folds}.log')
    orig_stdout, orig_stderr, sys.stdout, sys.stderr = init_logger(sys, log_filename, timestamp=True, verbose=True)
    print(f'Logged to file: {log_filename}')

    # Read model_module
    MM = exe.load_model_module(MODEL_NAME + "_module", os.path.join(PATH_TO_ROOT, MODEL_MODULE_PATH))

    # Print information
    print('Executed with arguments:')
    print(MM.ARGS)
    print('-' * 80)

    # Read dataset
    path_to_data = os.path.join(PATH_TO_ROOT, DATA_DIR)
    dset_df = MM.get_dset(path_to_data)

    # Add folds
    dset_df = exe.load_kfolds(dset_df, os.path.join(PATH_TO_ROOT, DATA_DIR, KFOLDS_FILE))

    # Get list of fold_ids
    fold_ids = dset_df.fold.dropna().unique().tolist()
    fold_ids.sort()

    print('-' * 80)
    print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

    # Folds to train
    train_folds = fold_ids
    if folds is not None:
        train_folds = [s for s in train_folds if s in folds]

    # Iterate folds
    for fold_id in train_folds:
        fold_id = [fold_id, ]

        print('-' * 40)
        offold_ids = [s for s in fold_ids if s not in fold_id]
        print(f"TRAINING FOLDS: '{','.join(offold_ids)}' TO PREDICT FOLD '{','.join(fold_id)}'")

        # Generate datasets
        datasets = {
            'train': dset_df[dset_df.train & ~dset_df.fold.isin(fold_id)],
            'valid': dset_df[dset_df.train & dset_df.fold.isin(fold_id) & dset_df.for_validation],
            'fold': dset_df[dset_df.train & dset_df.fold.isin(fold_id)],
            'test': dset_df[dset_df.test],
        }
        if debug:
            datasets['train'] = datasets['train'][0:len(datasets['valid'])]
        print(
            f"Training: {len(datasets['train']):,} | "
            f"Validation: {len(datasets['valid']):,} | "
            f"OOT-Fold: {len(datasets['fold']):,} | "
            f"Test: {len(datasets['test']):,}")

        # Get data loaders
        data_loaders = MM.get_dataloaders(path_to_data, datasets)

        # Get learner
        learner = MM.get_learner(nb_classes=None)

        # Train
        print('-' * 40 + " Training")
        epochs = MM.args.max_train_epochs if not debug else 1
        learner.train_loader(data_loaders, epochs=epochs, validation_metric=rmse)

        # Output_name
        ouput_name = MM.args.ouput_name if not debug else 'debug_' + MM.args.ouput_name

        # Save model
        filename = os.path.join(PATH_TO_ROOT, MODELS_PATH, ouput_name + f"_model_{','.join(fold_id)}.tar")
        MM.save_model(filename, learner)

        # Predict Fold
        print('-' * 40 + " Predicting Fold")
        valid_preds = learner.predict_loader(data_loaders['fold'])
        valid_preds_df = pd.DataFrame(valid_preds, index=datasets['fold'].image_id)
        valid_preds_df.columns = ['wind_speed']
        print(f"Dataset shape: {valid_preds_df.shape}")
        print(valid_preds_df.head(2))
        filepath = os.path.join(PATH_TO_ROOT, OUTPUTS_PATH, ouput_name + f"_fold_{','.join(fold_id)}.csv.gz")
        print(f"Saving FOLD predictions: {filepath}")
        valid_preds_df.to_csv(filepath, index=True)

        # Predict Test
        print('-' * 40 + " Predicting Test")
        test_preds = learner.predict_loader(data_loaders['test'])
        test_preds_df = pd.DataFrame(test_preds, index=datasets['test'].image_id)
        test_preds_df.columns = ['wind_speed']
        print(f"Dataset shape: {test_preds_df.shape}")
        test_preds_df.head(2)
        filepath = os.path.join(PATH_TO_ROOT, OUTPUTS_PATH, ouput_name + f"_test_{','.join(fold_id)}.csv.gz")
        print(f"Saving TEST predictions: {filepath}")
        test_preds_df.to_csv(filepath, index=True)

        learner.clean_memory()
        del learner


if __name__ == '__main__':
    main()
