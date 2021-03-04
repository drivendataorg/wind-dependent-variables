#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import argparse
import copy
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import datetime
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from dain import DAIN_Layer


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_cnn_idx', help='input cnn model predictions index [0-14]', required=True, type=int)
args = parser.parse_args()

# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[3]:


def train_model_snapshot(model, criterion, eval_criterion, lr, dataloaders, dataset_sizes, device, num_cycles, num_epochs_per_cycle):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000000.0
    model_w_arr = []
    for cycle in range(num_cycles):
        #initialize optimizer and scheduler each cycle
        #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam([{'params': model.conv.parameters()},
                                {'params': model.encoder_layer.parameters()},
                                {'params': model.transformer_encoder.parameters()},
                                {'params': model.linear.parameters()},
                                {'params': model.dean.mean_layer.parameters(), 'lr': lr * model.dean.mean_lr},
                                {'params': model.dean.scaling_layer.parameters(), 'lr': lr * model.dean.scale_lr},
                                {'params': model.dean.gating_layer.parameters(), 'lr': lr * model.dean.gate_lr},
                                ], lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs_per_cycle*len(dataloaders['train']))
        for epoch in range(num_epochs_per_cycle):
            print('Cycle {}: Epoch {}/{}'.format(cycle, epoch, num_epochs_per_cycle - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, img_inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    img_inputs = img_inputs.to(device)
                    labels = labels.to(device)
                    
                    org_p = torch.clone(inputs[:,0,-1:])
                    inputs = inputs[:,:,:-1]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, img_inputs)
                        loss = criterion(outputs, labels.reshape(-1,1))
                        eval_loss = eval_criterion(org_p + outputs, org_p + labels.reshape(-1,1))
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    
                    # statistics
                    running_loss += eval_loss.item() * inputs.size(0)

                epoch_loss = np.sqrt(running_loss / dataset_sizes[phase])

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            print()
        # deep copy snapshot
        model_w_arr.append(copy.deepcopy(model.state_dict()))

    ensemble_loss = 0.0

    #predict on validation using snapshots
    for inputs, img_inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        img_inputs = img_inputs.to(device)
        labels = labels.to(device)
        
        org_p = torch.clone(inputs[:,0,-1:])
        inputs = inputs[:,:,:-1]

        # forward
        # track history if only in train
        pred = torch.zeros((inputs.shape[0], 1), dtype = torch.float32).to(device)
        for weights in model_w_arr:
            model.load_state_dict(weights)
            model.eval()
            outputs = model(inputs, img_inputs) + org_p
            pred += outputs
        
        pred /= num_cycles
        eval_loss = eval_criterion(pred, org_p + labels.reshape(-1,1))
        ensemble_loss += eval_loss.item() * inputs.size(0)
    
    ensemble_loss /= dataset_sizes['val']
    ensemble_loss = np.sqrt(ensemble_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Ensemble Loss : {:4f}, Best val Loss: {:4f}'.format(ensemble_loss, best_loss))
    
    return model_w_arr, ensemble_loss, best_loss


# In[4]:


def test(model, models_w_arr, loader, n_imgs, device):
    res = np.zeros((n_imgs, 1), dtype = np.float32)
    for weights in models_w_arr:
        model.load_state_dict(weights) 
        model.eval()
        res_arr = []
        for inputs, img_inputs, _ in loader:
            inputs = inputs.to(device)
            img_inputs = img_inputs.to(device)
            org_p = torch.clone(inputs[:,0,-1:])
            #print(org_p)
            inputs = inputs[:,:,:-1]
            # forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs, img_inputs) + org_p
                res_arr.append(outputs.detach().cpu().numpy())
        res_arr = np.concatenate(res_arr, axis = 0)
        
        res += res_arr
    return res / len(models_w_arr)


# In[5]:


class WindDataset(Dataset):
    def __init__(self, features, img_features, gts, split_type):
        self.features = features
        self.img_features = img_features
        self.gts = gts
        
        #for c in range(means.shape[0]):
        #    self.features[:,:-1,c] = (self.features[:,:-1,c] - means[c]) / stds[c]
        #    self.features[:,:-1,c] = 2 * self.features[:,:-1,c] - 1
        
        self.split_type = split_type
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].transpose(1,0)
        img_feat = self.img_features[idx]
            
        feat = torch.FloatTensor(feat)
        img_feat = torch.FloatTensor(img_feat)
        if self.split_type == 'test':
            return feat, img_feat, -1
        return feat, img_feat, self.gts[idx]


# In[6]:


class TNet(nn.Module):
    def __init__(self, in_ch, n_outputs, p=0.05):
        super(TNet, self).__init__()
        
        self.dean = DAIN_Layer(input_dim = in_ch-1, mode='adaptive_scale', mean_lr=0.0001, gate_lr=0.01, scale_lr=0.001)
        
        self.conv = nn.Sequential(nn.Conv1d(in_ch, 64, 5, 1, 2), nn.ReLU())
        self.encoder_layer = nn.TransformerEncoderLayer(64, 8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 1)
        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p*2)
        self.linear = nn.Linear(64 + 2048, n_outputs)

    def forward(self, x, x_img):
        sh = x.shape
        x[:,:2] = self.dean(x[:,:2])
        #x_last = x[:,:,-1]
        x = self.dropout(x)
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = self.avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        x_img = self.dropout2(x_img)
        x = torch.cat([x, x_img], dim = 1)
        #x = self.fc1(x)
        x = self.linear(x)
        return x


# In[7]:


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


# In[8]:


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


# In[9]:


img_inp_path_arr = ['resnet50_models', '3ts_imgs_resnet50_models_10folds']


# In[10]:


img_out_path_arr = ['res50', '3ts_imgs_res50_10folds']


# In[11]:


path_idx = args.input_cnn_idx
img_path_idx = 0


# In[12]:


time_steps = 25


# In[13]:


train_df = pd.read_csv(inp_path_arr[path_idx]+'/train.csv')
train_df.set_index("image_id", inplace=True)
train_df['split'] = 'train'

test_df = pd.read_csv(inp_path_arr[path_idx]+'/test.csv')
test_df.set_index("image_id", inplace=True)
test_df['split'] = 'test'
test_df['wind_speed'] = -1

df = pd.concat([train_df, test_df], axis = 0)


# In[14]:


train_img_features = np.load(img_inp_path_arr[img_path_idx]+'/train_features.npy')
test_img_features = np.load(img_inp_path_arr[img_path_idx]+'/test_features.npy')


# In[15]:


#img_features = np.concatenate([train_img_features, test_img_features], 0)


# In[16]:


train_img_features.shape


# In[17]:


train_feats = []
train_targets = []
train_org_pred = []
train_storm_ids = []
train_cum_count = []
test_feats = []
test_org_pred = []
test_storm_ids = []
test_img_ids = []


# In[18]:


for _, storm_df in df.groupby('storm_id'):
    storm_df = storm_df.sort_values('relative_time')
    
    n = storm_df.shape[0]
    feats_arr = np.zeros((n+time_steps, 3), dtype=np.float32)
    
    for i in range(n):
        split = storm_df['split'].iloc[i]
        target = storm_df['wind_speed'].iloc[i]
        storm_id = storm_df['storm_id'].iloc[i]
        img_id = storm_df.index[i]
        p = storm_df['pred'].iloc[i]
        oc = storm_df['ocean'].iloc[i]
        t = storm_df['relative_time'].iloc[i]
        
        feats_arr[i+time_steps,0] = p
        feats_arr[i+time_steps,1] = t
        feats_arr[i+time_steps,2] = oc
        
        temp = feats_arr[i+1: i+time_steps+1].copy()
        if i < time_steps:
            temp[:(time_steps-i)] = temp[time_steps-i-1]
        temp[:-1,0] -= temp[-1:,0]
        temp[:-1,1] -= temp[-1:,1]
        #temp[feats_arr[i+1: i+time_steps+1] == 0] = 0
        
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


# In[19]:


train_feats = np.array(train_feats)
train_targets = np.array(train_targets).astype(np.float32)
train_org_pred = np.array(train_org_pred)
train_storm_ids = np.array(train_storm_ids)
train_cum_count = np.array(train_cum_count)
test_feats = np.array(test_feats)
test_storm_ids = np.array(test_storm_ids)
test_org_pred = np.array(test_org_pred)


# In[20]:


#train_img_features = img_features[:train_img_features.shape[0]]
#test_img_features = img_features[train_img_features.shape[0]:]


# In[21]:


test_final_pred = np.zeros_like(test_org_pred)


# In[22]:


group_kfold = GroupKFold(n_splits=5)


# In[23]:


models_arr = []
val_pred = np.zeros_like(train_targets)
sc_arr = []
fold = 0
for train_index, val_index in group_kfold.split(train_feats, train_targets, train_storm_ids):
    print(fold)
    fold += 1
    image_datasets = {'train': WindDataset(train_feats[train_index], train_img_features[train_index], train_targets[train_index], 'train'),
                      'val': WindDataset(train_feats[val_index], train_img_features[val_index], train_targets[val_index], 'val'),
                      'test': WindDataset(test_feats, test_img_features, None, 'test')}

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=8),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=256, shuffle=False, num_workers=8),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=256, shuffle=False, num_workers=8)}

    model = TNet(train_feats.shape[-1], 1)
    model = model.to(device)

    criterion = nn.MSELoss()
    eval_criterion = nn.MSELoss()

    dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}

    #train a model on this data split using snapshot ensemble
    model_ft_arr, sc, _ = train_model_snapshot(model, criterion, eval_criterion, 0.004, dataloaders, dataset_sizes, device,
                           num_cycles=1, num_epochs_per_cycle=6)
    models_arr.extend(model_ft_arr)
    sc_arr.append(sc)
    test_final_pred += test(model, model_ft_arr, dataloaders['test'], test_final_pred.shape[0], device)[:,0]
    val_pred[val_index] += test(model, model_ft_arr, dataloaders['val'], train_feats[val_index].shape[0], device)[:,0]
    #break
print(np.mean(sc_arr))
test_final_pred /= 5


# In[24]:


sub = pd.read_csv('submission_format.csv')
sub['image_id'] = test_img_ids
sub['wind_speed'] = np.round(test_final_pred).astype(np.int64)
sub.to_csv('transformer_dain_%dts_after_%s_and_img_feat_%s.csv'%(time_steps, out_path_arr[path_idx], img_out_path_arr[img_path_idx]), index = False)
np.save('transformer_dain_%dts_after_%s_and_img_feat_%s'%(time_steps, out_path_arr[path_idx], img_out_path_arr[img_path_idx]), test_final_pred)
np.save('transformer_dain_%dts_after_%s_and_img_feat_%s_val'%(time_steps, out_path_arr[path_idx], img_out_path_arr[img_path_idx]), val_pred)
np.save('sub_test_img_ids', test_img_ids)


# In[ ]:




