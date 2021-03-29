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
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import random
from sklearn.feature_selection import SelectKBest, f_regression


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_cnn_idx', help='input cnn model predictions index [0-14]', required=True, type=int)
args = parser.parse_args()

# In[2]:


time_steps = 30
def set_seed():
    np.random.seed(4321)
    random.seed(4321)
set_seed()


# In[3]:


inp_path_arr = ['6ts_imgs_time_aug_resnet50_models',
            '6ts_imgs_time_aug_resnet50_models_021',
            '6ts_imgs_time_aug_resnet50_models_102',
            '6ts_imgs_time_aug_resnet50_models_120',
            '6ts_imgs_time_aug_resnet50_models_201',
            '6ts_imgs_time_aug_resnet50_models_210',
            '3ts_imgs_resnet50_models_10folds',
            'resnet50_models',
            '12ts_imgs_resnet18_models',
            '9ts_imgs_time_aug_resnet50_models', 
            '6ts_2space_imgs_time_aug_resnet50_models', 
            '6ts_3space_imgs_time_aug_resnet50_models', 
            '6ts_4space_imgs_time_aug_resnet50_models', 
            '6ts_5space_imgs_time_aug_resnet50_models', 
            '6ts_6space_imgs_time_aug_resnet50_models']


# In[4]:


out_path_arr = ['6ts_imgs_time_aug_res50',
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


# In[5]:


path_idx = args.input_cnn_idx


# In[6]:


train_df = pd.read_csv(inp_path_arr[path_idx]+'/train.csv')
#train_df.set_index("image_id", inplace=True)
train_df['split'] = 'train'

test_df = pd.read_csv(inp_path_arr[path_idx]+'/test.csv')
#test_df.set_index("image_id", inplace=True)
test_df['split'] = 'test'
test_df['wind_speed'] = -1

df = pd.concat([train_df, test_df], axis = 0)
df.reset_index(inplace=True)


# In[7]:


df


# In[8]:


train_img_features = np.load('resnet50_models/train_features.npy')
test_img_features = np.load('resnet50_models/test_features.npy')


# In[9]:


selected_feat_idx = np.argsort(train_img_features.std(0))[-10:]


# In[10]:


img_features = np.concatenate([train_img_features, test_img_features], 0)[:, selected_feat_idx]


# In[11]:


train_feats = []
train_img_feats = []
train_targets = []
train_org_pred = []
train_storm_ids = []
train_cum_count = []
test_feats = []
test_img_feats = []
test_org_pred = []
test_storm_ids = []
test_img_ids = []


# In[12]:


for _, storm_df in df.groupby('storm_id'):
    storm_df = storm_df.sort_values('relative_time')
    
    n = storm_df.shape[0]
    feats_arr = np.zeros((n+time_steps, 5), dtype=np.float32)
    
    for i in range(n):
        split = storm_df['split'].iloc[i]
        target = storm_df['wind_speed'].iloc[i]
        storm_id = storm_df['storm_id'].iloc[i]
        img_id = storm_df['image_id'].iloc[i]
        p = storm_df['pred'].iloc[i]
        oc = storm_df['ocean'].iloc[i]
        t = storm_df['relative_time'].iloc[i]
        idx = storm_df.index[i]
        
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
            train_img_feats.append(img_features[idx])
        else:
            test_feats.append(temp)
            test_storm_ids.append(storm_id)
            test_img_ids.append(img_id)
            test_org_pred.append(p)
            test_img_feats.append(img_features[idx])


# In[13]:


train_feats = np.array(train_feats)
train_img_feats = np.array(train_img_feats)
train_targets = np.array(train_targets)
train_org_pred = np.array(train_org_pred)
train_storm_ids = np.array(train_storm_ids)
train_cum_count = np.array(train_cum_count)
test_feats = np.array(test_feats)
test_img_feats = np.array(test_img_feats)
test_storm_ids = np.array(test_storm_ids)
test_org_pred = np.array(test_org_pred)


# In[14]:


#sel = SelectKBest(f_regression, k=10)
#train_img_feats = sel.fit_transform(train_img_feats, train_targets)
#test_img_feats = sel.transform(test_img_feats)


# In[15]:


def generate_features(feats, img_feats):
    n = feats.shape[0]
    feats = np.concatenate([feats[:,:,0], feats[:,-1:,1], feats[:,:-1,2], feats[:,:-1,3], feats[:,-6:,4], feats[:,:-1,2:3].sum(1), feats[:,:-1,0:1].sum(1)], axis = 1)
    #feats[feats == 100] = -100
    return feats


# In[16]:


train_feats = generate_features(train_feats, train_img_feats)
test_feats = generate_features(test_feats, test_img_feats)


# In[17]:


best = {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 331, 'min_samples_split': 136.0, 'splitter': 'random'}


# In[18]:


group_kfold = GroupKFold(n_splits=5)


# In[19]:


val_pred = np.zeros((train_feats.shape[0],), dtype = np.float32)
test_pred = np.zeros((test_feats.shape[0],), dtype = np.float32)


# In[20]:


#set_seed()
fold = 0
error = 0
orig_error = 0
for train_index, val_index in group_kfold.split(train_feats, train_targets, train_storm_ids):
    print(fold)
    fold += 1
    
    model = DecisionTreeRegressor(max_depth = int(best['max_depth']),
                                  min_samples_split = int(best['min_samples_split']),
                                  min_samples_leaf = int(best['min_samples_leaf']),
                                  max_features = best['max_features'],
                                  splitter = best['splitter'],
                                  random_state = 4321)
    model.fit(train_feats[train_index], train_targets[train_index])
    val_pred[val_index] = model.predict(train_feats[val_index]) + train_org_pred[val_index]
    test_pred += model.predict(test_feats) + test_org_pred
    fold_val_error = np.sqrt(mean_squared_error(train_targets[val_index] + train_org_pred[val_index], np.round(val_pred[val_index])))
    error += fold_val_error
    fold_orig_error = np.sqrt(mean_squared_error(train_targets[val_index] + train_org_pred[val_index], train_org_pred[val_index]))
    orig_error += fold_orig_error
    print(fold_val_error, fold_orig_error)
    
print('total error', error/fold, orig_error/fold)
test_pred /= fold


# In[21]:


sub = pd.read_csv('submission_format.csv')
sub['image_id'] = test_img_ids
sub['wind_speed'] = np.round(test_pred).astype(np.int64)
sub.to_csv('dt_%dts_after_%s.csv'%(time_steps, out_path_arr[path_idx]), index = False)
np.save('dt_%dts_after_%s'%(time_steps, out_path_arr[path_idx]), test_pred)
np.save('dt_%dts_after_%s_val'%(time_steps, out_path_arr[path_idx]), val_pred)
#np.save('sub_test_img_ids', test_img_ids)


# In[ ]:




