import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import argparse
import copy
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error
import datetime
from PIL import Image
from tqdm import tqdm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import random
import itertools

def rmse(y, p):
    return np.sqrt(mean_squared_error(y,np.round(p)))

class ensembleSelectionArithmetic:

    def __init__(self, metric):
        self.metric = metric
        
    def _compare(self, sc1, sc2):
        if sc1 < sc2:
            return True
        return False
        
    def _initialize(self, X_p, y):
        current_sc = self.metric(y, X_p[0])
        ind = 0
        for i in range(1, X_p.shape[0]):
            sc = self.metric(y, X_p[i])
            if self._compare(sc, current_sc):
                current_sc = sc
                ind = i
        return ind, current_sc
        
    def es_with_replacement(self, X_p, Xtest_p, y):
        best_ind, best_sc = self._initialize(X_p, y)
        current_sc = best_sc
        sumP = np.copy(X_p[best_ind])
        sumP_test = np.copy(Xtest_p[best_ind])
        i = 1
        while True:
            i += 1
            ind = -1
            for m in range(X_p.shape[0]):
                sc = self.metric(y, (sumP+X_p[m])/i)
                if self._compare(sc, current_sc):
                    current_sc = sc
                    ind = m
            if ind>-1:
                sumP += X_p[ind]
                sumP_test += Xtest_p[ind]
            else:
                break
        sumP /= (i-1)
        sumP_test /= (i-1)
        
        return current_sc, sumP, sumP_test
        
    def es_with_bagging(self, X_p, Xtest_p, y, f = 0.5, n_bags = 20):
        list_of_indecies = [i for i in range(X_p.shape[0])]
        bag_size = int(f*X_p.shape[0])
        sumP = None
        sumP_test = None
        for i in range(n_bags):
            model_weight = [0 for j in range(X_p.shape[0])]
            rng = np.copy(list_of_indecies)
            np.random.shuffle(rng)
            rng = rng[:bag_size]
            sc, p, ptest = self.es_with_replacement(X_p[rng], Xtest_p[rng], y)
            print('bag: %d, sc: %f'%(i, sc))
            if sumP is None:
                sumP = p
                sumP_test = ptest
            else:
                sumP += p
                sumP_test += ptest
                
        sumP /= n_bags
        sumP_test /= n_bags
        final_sc = self.metric(y, sumP)
        print('avg sc: %f'%(final_sc))
        return (final_sc, sumP, sumP_test)

np.random.seed(4321)
random.seed(4321)

train_df = pd.read_csv('6ts_imgs_time_aug_resnet50_models_210/train.csv')
#train_df.set_index("image_id", inplace=True)

train_targets = []
train_storm_ids = []
train_img_id = []
for _, storm_df in train_df.groupby('storm_id'):
    storm_df = storm_df.sort_values('relative_time')
    train_targets.append(storm_df['wind_speed'].values)
    train_storm_ids.extend(storm_df['storm_id'].values.tolist())
    train_img_id.extend(storm_df['image_id'].values.tolist())

train_targets = np.concatenate(train_targets, 0)

val_pred_arr = list(np.load('xgb_bulldozer_100trails_v3.npy')) 
test_pred_arr = list(np.load('xgb_bulldozer_100trails_test_v3.npy'))

ext_arr = ['6ts_imgs_time_aug_res50',
            '6ts_imgs_time_aug_res50_021',
            '6ts_imgs_time_aug_res50_102',
            '6ts_imgs_time_aug_res50_120',
            '6ts_imgs_time_aug_res50_201',
            '6ts_imgs_time_aug_res50_210',
            '3ts_imgs_res50_10folds',
            'res50',
            '12ts_imgs_res18',
            '9ts_imgs_time_aug_res50',
            '6ts_2space_imgs_time_aug_res50',
            '6ts_3space_imgs_time_aug_res50',
            '6ts_4space_imgs_time_aug_res50',
            '6ts_5space_imgs_time_aug_res50',
            '6ts_6space_imgs_time_aug_res50']

ts_cnn_paths = ['ts_cnn_dain_25ts_after_%s_and_img_feat_res50'%(ext) for ext in ext_arr]

for path in ts_cnn_paths:
    val_pred_arr.append(np.load(path+'_val.npy'))
    test_pred_arr.append(np.load(path+'.npy'))

mlp_paths = ['mlp_dain_25ts_after_%s_and_img_feat_res50'%(ext) for ext in ext_arr]

for path in mlp_paths:
    val_pred_arr.append(np.load(path+'_val.npy'))
    test_pred_arr.append(np.load(path+'.npy'))

lstm_paths = ['lstm_dain_25ts_after_%s_and_img_feat_res50'%(ext) for ext in ext_arr]

for path in lstm_paths:
    val_pred_arr.append(np.load(path+'_val.npy'))
    test_pred_arr.append(np.load(path+'.npy'))

gru_paths = ['gru_dain_25ts_after_%s_and_img_feat_res50'%(ext) for ext in ext_arr]

for path in gru_paths:
    val_pred_arr.append(np.load(path+'_val.npy'))
    test_pred_arr.append(np.load(path+'.npy'))

transformer_paths = ['transformer_dain_25ts_after_%s_and_img_feat_res50'%(ext) for ext in ext_arr]

for path in transformer_paths:
    val_pred_arr.append(np.load(path+'_val.npy'))
    test_pred_arr.append(np.load(path+'.npy'))

linear_paths = ['linear_dain_25ts_after_%s_and_img_feat_res50'%(ext) for ext in ext_arr]

for path in linear_paths[:6] + linear_paths[10:]:
    val_pred_arr.append(np.load(path+'_val.npy'))
    test_pred_arr.append(np.load(path+'.npy'))

dt_paths = ['dt_30ts_after_%s'%(ext) for ext in ext_arr]

for path in dt_paths[:6] + dt_paths[10:]:
    val_pred_arr.append(np.load(path+'_val.npy'))
    test_pred_arr.append(np.load(path+'.npy'))

val_pred_arr = np.array(val_pred_arr)
test_pred_arr = np.array(test_pred_arr)

print(val_pred_arr.shape, test_pred_arr.shape)

es_obj = ensembleSelectionArithmetic(rmse)
sc, es_val_pred, es_test_pred = es_obj.es_with_bagging(val_pred_arr, test_pred_arr, train_targets, n_bags = 20, f = 0.75)

print(rmse(train_targets, es_val_pred**0.995))

ids = np.load('sub_test_img_ids.npy')

sub = pd.read_csv('submission_format.csv')
sub['image_id'] = ids
sub['wind_speed'] = np.round(es_test_pred**0.995).astype(np.int64)
sub.to_csv('final_sub.csv', index = False)
