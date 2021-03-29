#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
import itertools


# In[2]:


np.random.seed(4321)
random.seed(4321)


# In[3]:


def reshape_features(feats):
    n = feats.shape[0]
    feats = np.concatenate([feats[:,:,0], feats[:,-1:,1], feats[:,:-1,2], feats[:,:-1,3], feats[:,-6:,4], feats[:,:-1,2:3].sum(1), feats[:,:-1,0:1].sum(1)], axis = 1)
    #feats[feats == 100] = -100
    return feats


# In[4]:


def generate_features(path, time_steps):
    train_df = pd.read_csv(path+'/train.csv')
    train_df.set_index("image_id", inplace=True)
    train_df['split'] = 'train'

    test_df = pd.read_csv(path+'/test.csv')
    test_df.set_index("image_id", inplace=True)
    test_df['split'] = 'test'
    test_df['wind_speed'] = -1

    df = pd.concat([train_df, test_df], axis = 0)
    
    train_feats = []
    train_targets = []
    train_org_pred = []
    train_storm_ids = []
    train_cum_count = []
    test_feats = []
    test_org_pred = []
    test_storm_ids = []
    test_img_ids = []
    
    for _, storm_df in df.groupby('storm_id'):
        storm_df = storm_df.sort_values('relative_time')

        n = storm_df.shape[0]
        feats_arr = np.zeros((n+time_steps, 5), dtype=np.float32)

        for i in range(n):
            split = storm_df['split'].iloc[i]
            target = storm_df['wind_speed'].iloc[i]
            storm_id = storm_df['storm_id'].iloc[i]
            #img_id = storm_df['image_id'].iloc[i]
            p = storm_df['pred'].iloc[i]
            oc = storm_df['ocean'].iloc[i]
            t = storm_df['relative_time'].iloc[i]
            img_id = storm_df.index[i]

            feats_arr[i+time_steps,0] = p
            feats_arr[i+time_steps,1] = oc
            feats_arr[i+time_steps,2] = t
            feats_arr[i+time_steps,3] = t
            feats_arr[i+time_steps,4] = p

            temp = feats_arr[i+1: i+time_steps+1].copy()
            if i < time_steps:
                temp[:(time_steps-i)] = temp[time_steps-i-1]
            temp[:-1, 2] = np.abs(temp[1:, 0] - temp[:-1, 0]) / np.abs(temp[1:, 2] - temp[:-1, 2] + 1e-6)
            temp[:-1,0] -= temp[-1:,0]

            if i == 0:
                temp[-6:,4] = np.array([temp[feats_arr[i+1: i+time_steps+1,0] > 0,4].std(), 
                                        temp[feats_arr[i+1: i+time_steps+1,0] > 0,4].max(),
                                        (t - temp[np.argmax(temp[:,4]), 2])*np.sign(p - np.max(temp[:,4])),
                                        1000,1000,1000])
            else:
                temp[-6:,4] = np.array([temp[feats_arr[i+1: i+time_steps+1,0] > 0,4].std(), 
                                        temp[feats_arr[i+1: i+time_steps+1,0] > 0,4].max(),
                                        (t - temp[np.argmax(temp[:,4]), 2])*np.sign(p - np.max(temp[:,4])),
                                        temp[:-1][feats_arr[i+1: i+time_steps,0] > 0,0].max(),
                                        temp[:-1][feats_arr[i+1: i+time_steps,0] > 0,0].min(),
                                        temp[:-1][feats_arr[i+1: i+time_steps,0] > 0,2].std()])
            #temp[feats_arr[i+1: i+time_steps+1,0] == 0, 0] = -100
            temp[:-1,3] -= temp[-1:,3]

            if split == 'train':
                train_feats.append(temp)
                train_targets.append(target-p)
                train_org_pred.append(p)
                train_storm_ids.append(storm_id)
                train_cum_count.append(i/n)
            else:
                test_feats.append(temp)
                test_storm_ids.append(storm_id)
                test_img_ids.append(img_id)
                test_org_pred.append(p)
    
    train_feats = np.array(train_feats)
    train_targets = np.array(train_targets)
    train_org_pred = np.array(train_org_pred)
    train_storm_ids = np.array(train_storm_ids)
    train_cum_count = np.array(train_cum_count)
    test_feats = np.array(test_feats)
    test_storm_ids = np.array(test_storm_ids)
    test_org_pred = np.array(test_org_pred)
    
    train_feats = reshape_features(train_feats)
    test_feats = reshape_features(test_feats)
    
    return train_feats,  train_targets, train_org_pred, train_storm_ids, test_feats, test_org_pred, test_storm_ids


# In[5]:


group_kfold = GroupKFold(n_splits=5)


# In[6]:


path_arr = ['6ts_imgs_time_aug_resnet50_models',
            '6ts_imgs_time_aug_resnet50_models_021',
            '6ts_imgs_time_aug_resnet50_models_102',
            '6ts_imgs_time_aug_resnet50_models_120',
            '6ts_imgs_time_aug_resnet50_models_201',
            '6ts_imgs_time_aug_resnet50_models_210',
            '3ts_imgs_resnet50_models_10folds',
            'resnet50_models',
            '12ts_imgs_resnet18_models',
            '9ts_imgs_time_aug_resnet50_models']


# In[7]:


val_pred_arr = []
test_pred_arr = []


# In[8]:


def objective(space):
    global idx, val_pred_arr, test_pred_arr
    error = 0
    train_feats, train_targets, train_org_pred, train_storm_ids, test_feats, test_org_pred, test_storm_ids = generate_features(path_arr[space['path_idx']], space['time_steps'])
    
    val_pred = np.zeros((train_feats.shape[0],), dtype = np.float32)
    test_pred = np.zeros((test_feats.shape[0],), dtype = np.float32)
    
    for train_index, val_index in group_kfold.split(train_feats, train_targets, train_storm_ids):
        model = xgb.XGBRegressor(n_estimators = space['n_estimators'],
                                max_depth = int(space['max_depth']),
                                learning_rate = space['learning_rate'],
                                gamma = space['gamma'],
                                min_child_weight = space['min_child_weight'],
                                subsample = space['subsample'],
                                colsample_bytree = space['colsample_bytree'],
                                n_jobs = 8,
                                random_state = 4321)
        model.fit(train_feats[train_index], train_targets[train_index])
        val_pred[val_index] = model.predict(train_feats[val_index]) + train_org_pred[val_index]
        test_pred += model.predict(test_feats) + test_org_pred
        error += np.sqrt(mean_squared_error(train_targets[val_index] + train_org_pred[val_index], val_pred[val_index]))
        
    error /= 5
    print(space, error)
    test_pred /= 5
    val_pred_arr.append(val_pred)
    test_pred_arr.append(test_pred)
    return {'loss': error, 'status': STATUS_OK }


# In[9]:


space = {
    'max_depth' : hp.choice('max_depth', range(3, 10, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 200, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.5, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01),
    'path_idx' : hp.choice('path_idx', range(len(path_arr))),
    'time_steps' : hp.choice('time_steps', range(7, 40)),
}


# In[10]:


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            rstate= np.random.seed(4321))


# In[11]:


np.save('xgb_bulldozer_100trails_v3', np.array(val_pred_arr))
np.save('xgb_bulldozer_100trails_test_v3', np.array(test_pred_arr))


# In[ ]:




