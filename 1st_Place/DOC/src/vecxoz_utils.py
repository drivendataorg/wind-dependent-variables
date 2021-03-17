#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ArgumentParserExtended(ArgumentParser):
    """
    The main purpose of this class is to standardize and simplify definition of arguments
    and allow processing of True, False, and None values.
    There are 4 types of arguments (bool, int, float, str). All accept None.
    
    Usage:

    parser = ArgumentParserExtended()
    
    parser.add_str('--str', default='/home/user/data')
    parser.add_int('--int', default=220)
    parser.add_float('--float', default=3.58)
    parser.add_bool('--bool', default=True)
    
    args = parser.parse_args()
    print(parser.args_repr(args, True))
    """

    def __init__(self, *args, **kwargs):
        super(ArgumentParserExtended, self).__init__(*args, **kwargs)

    def bool_none_type(self, x):
        if x == 'True':
            return True
        elif x == 'False':
            return False
        elif x == 'None':
            return None
        else:
            raise ValueError('Unexpected literal for bool type')

    def int_none_type(self, x):
        return None if x == 'None' else int(x)

    def float_none_type(self, x):
        return None if x == 'None' else float(x)

    def str_none_type(self, x):
        return None if x == 'None' else str(x)

    def add_str(self, name, default=None, choices=None, help='str or None'):
        """
        Returns str or None
        """
        _ = self.add_argument(name, type=self.str_none_type, default=default, choices=choices, help=help)

    def add_int(self, name, default=None, choices=None, help='int or None'):
        """
        Returns int or None
        'hello' or 'none' or 1.2 will cause an error
        """
        _ = self.add_argument(name, type=self.int_none_type, default=default, choices=choices, help=help)

    def add_float(self, name, default=None, choices=None, help='float or None'):
        """
        Returns float or None
        'hello' or 'none' will cause an error
        """
        _ = self.add_argument(name, type=self.float_none_type, default=default, choices=choices, help=help)

    def add_bool(self, name, default=None, help='bool'):
        """
        Returns True, False, or None
        Anything except 'True' or 'False' or 'None' will cause an error

        `choices` are checked after type conversion of argument passed in fact
            i.e. `choices` value must be True instead of 'True'
        Default value is NOT checked using `choices`
        Default value is NOT converted using `type`
        """
        _ = self.add_argument(name, type=self.bool_none_type, default=default, choices=[True, False, None], help=help)

    @staticmethod
    def args_repr(args, print_types=False):
        ret = ''
        props = vars(args)
        keys = sorted([key for key in props])
        vals = [str(props[key]) for key in props]
        max_len_key = len(max(keys, key=len))
        max_len_val = len(max(vals, key=len))
        if print_types:
            for key in keys:
                ret += '%-*s  %-*s  %s\n' % (max_len_key, key, max_len_val, props[key], type(props[key]))
        else:   
            for key in keys:
                ret += '%-*s  %s\n' % (max_len_key, key, props[key])
        return ret.rstrip()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_tpu(tpu_ip_or_name=None):
    """
    Initializes `TPUStrategy` or appropriate alternative.
    IMPORTANT: Run this init before init of `tf.data`

    tpu_ip_or_name : str or None
        e.g. 'grpc://10.70.50.202:8470' or 'node-1'

    Usage:
    
    tpu, topology, strategy = init_tpu('node-1')
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_ip_or_name)
        tf.config.experimental_connect_to_cluster(tpu)
        topology = tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('--> Master:      ', tpu.master())
        print('--> Num replicas:', strategy.num_replicas_in_sync)
        return tpu, topology, strategy
    except:
        print('--> TPU was not found!')
        # strategy = tf.distribute.get_strategy() # CPU or single GPU
        strategy = tf.distribute.MirroredStrategy() # GPU or multi-GPU
        # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy() # clusters of multi-GPU
        print('--> Num replicas:', strategy.num_replicas_in_sync)
        return None, None, strategy

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_tfdata(files_glob, is_train, batch_size, auto, parse_example, 
                mod=None, aug=None, tta=None, norm=None, 
                buffer_size=2048, use_cache=True):
    """
    Creates tf.data.TFRecordDataset with appropriate train/test parameters depending on `is_train`.
    ** IMPORTANT 1: if we have 1) small RAM or 2) large data or 3) large example (image sequence):
        do not CACHE
        use small SHUFFLE buffer (e.g. 128) or disable SHUFFLE (shuffle before TFRecord creation)
    ** IMPORTANT 2:
        large SHUFFLE buffer can slow down training process

    files_glob : str
        Glob wildcard for TFRecord files
    is_train : bool
        if is_train == True:
            deterministic = False
            shuffle TFRec files
            apply AUG
            do NOT apply TTA
            repeat
            shuffle examples
            do NOT cache
    batch_size : int
    auto : int
    parse_example, mod, aug, tta, norm : callable
        Processing functions
    buffer_size : int or None
        Shuffle buffer size
    use_cache : bool
        Whether to cache data

    Example:
        train_ds = init_tfdata(train_glob, 
                               is_train=True,  
                               batch_size=args.batch_size, 
                               auto=args.auto,
                               parse_example=parse_example, 
                               aug=aug, 
                               norm=norm)
        val_ds = init_tfdata(val_glob, 
                             is_train=False,  
                             batch_size=args.batch_size, 
                             auto=args.auto,
                             parse_example=parse_example,
                             norm=norm)
    """
    options = tf.data.Options()
    options.experimental_deterministic = not is_train
    files = tf.data.Dataset.list_files(files_glob, shuffle=is_train).with_options(options)
    #
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=auto)
    ds = ds.with_options(options)
    ds = ds.map(parse_example, num_parallel_calls=auto)
    if mod is not None:
        ds = ds.map(mod, num_parallel_calls=auto)
    if is_train and aug is not None:
        ds = ds.map(aug, num_parallel_calls=auto)
    if not is_train and tta is not None:
        ds = ds.map(tta, num_parallel_calls=auto)
    if norm is not None:
        ds = ds.map(norm, num_parallel_calls=auto)
    if is_train:
        ds = ds.repeat()
        if buffer_size is not None:
            ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(auto)
    if not is_train and use_cache:
        ds = ds.cache()
    #
    return ds

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class KeepLastCKPT(tf.keras.callbacks.Callback):
    """
    Sort all ckpt files matching the wildcard and remove all except last.
    If there is only one ckpt file it will not be removed.
    If save_best_only=True in ModelCheckpoint and 
        naming is consistent e.g. "model-best-f0-e001-25.3676.h5"
        then KeepLastCKPT will keep OVERALL best ckpt
    
    NOTE:
    Methods `on_epoch_end` and `on_test_end` are called before last ckpt is created.
    Order of callbacks in the list passed to `model.fit` does not affect this behavor.
    """
    #
    def __init__(self, wildcard):
        super(KeepLastCKPT, self).__init__()
        self.wildcard = wildcard
    #
    def on_epoch_begin(self, epoch, logs=None):
        # files = sorted(glob.glob(self.wildcard))
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                # os.remove(file)
                tf.io.gfile.remove(file)
            print('Kept ckpt: %s' % files[-1])
        else:
            print('No ckpt to keep')
    #
    def on_train_end(self, logs=None):
        # files = sorted(glob.glob(self.wildcard))
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                # os.remove(file)
                tf.io.gfile.remove(file)
            print('\nKept ckpt (final): %s' % files[-1])
        else:
            print('\nNo ckpt to keep (final)')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def compute_cv_scores(data_dir, preds_dir, n_folds, tta_number, print_scores=True):
    # Load csv
    train_df = pd.read_csv(os.path.join(data_dir, 'train_cv.csv'))
    unique_labels_sorted = np.load(os.path.join(data_dir, 'unique_labels_sorted.npy'))
    
    # Collect all preds
    all_tta = []
    for tta_id in range(tta_number + 1):
        all_folds = []
        for fold_id in range(n_folds):
            all_folds.append(np.load(os.path.join(preds_dir, 'y_pred_val_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel())
        all_tta.append(np.hstack(all_folds))
    
    # Collect coresponding true label
    y_true_list = []
    for fold_id in range(n_folds):
        y_true_list.append(train_df.loc[train_df['fold_id'] == fold_id, 'wind_speed'].values.ravel())
    y_true = np.hstack(y_true_list)
    
    # Compute score for original image and each TTA
    scores = []
    for tta_id, y_pred in enumerate(all_tta):
        score = mean_squared_error(y_true, np.int32(np.round(y_pred)), squared=False)
        scores.append(score)
        if print_scores:
            print('TTA %d score: %.4f' % (tta_id, score))

    # Compute score for mean of all TTA
    score = mean_squared_error(y_true, np.int32(np.round(np.mean(all_tta, axis=0))), squared=False)
    scores.append(score)
    if print_scores:
        print('-------------------')
        print('MEAN of all: %.4f' % score)

    return scores

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_submission(data_dir, preds_dir, n_folds, tta_number, file_name=None):
    if file_name is None:
        file_name = os.getcwd().split('/')[-1][:17] + '.csv'
    # Load csv
    subm_df = pd.read_csv(os.path.join(data_dir, 'submission_format.csv'))

    # Collect test preds
    y_preds_test = []
    for tta_id in range(tta_number + 1):
        for fold_id in range(n_folds):
            y_preds_test.append(np.load(os.path.join(preds_dir, 'y_pred_test_fold_%d_tta_%d.npy' % (fold_id, tta_id))).ravel())
    
    # Write submission
    subm_df['wind_speed'] = np.int32(np.round(np.mean(y_preds_test, axis=0)))
    subm_df.to_csv(file_name, index=False)
    
    return file_name

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def convert_proba(data_dir, preds_dir, n_folds, tta_number):
    """ 
    Convert probabilities to predictions.
    Initially standard "preds" dir contains probas.
    We rename in to "preds_proba", then create "preds" and save converted result.
    """

    preds_dir_proba = preds_dir + '_proba'
    os.rename(preds_dir, preds_dir_proba)
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)

    train_df = pd.read_csv(os.path.join(data_dir, 'train_cv.csv'))
    unique_labels_sorted = np.sort(train_df['wind_speed'].unique())
    
    for tta_id in range(tta_number + 1):
        for fold_id in range(n_folds):
            for subset in ['val', 'test']:
                name = 'y_pred_%s_fold_%d_tta_%d.npy' % (subset, fold_id, tta_id)
                y_pred_proba = np.load(os.path.join(preds_dir_proba, name))
                y_pred_ind = np.argmax(y_pred_proba, axis=1)
                y_pred = unique_labels_sorted[y_pred_ind]
                np.save(os.path.join(preds_dir, name), y_pred)

    return preds_dir_proba

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

