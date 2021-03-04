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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import datetime
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from efficientnet_pytorch import EfficientNet


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
        optimizer = optim.Adam(model.parameters(), lr=lr)
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
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)                    

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.reshape(-1,1))
                        eval_loss = eval_criterion(outputs, labels.reshape(-1,1))
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
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        pred = torch.zeros((inputs.shape[0], 1), dtype = torch.float32).to(device)
        for weights in model_w_arr:
            model.load_state_dict(weights)
            model.eval()
            outputs = model(inputs)
            pred += outputs
        
        pred /= num_cycles
        eval_loss = eval_criterion(pred, labels.reshape(-1,1))
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
        for inputs, _ in loader:
            inputs = inputs.to(device)
            # forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs)    
                res_arr.append(outputs.detach().cpu().numpy())
        res_arr = np.concatenate(res_arr, axis = 0)
        
        res += res_arr
    return res / len(models_w_arr)


# In[5]:


def test_augment(model, models_w_arr, loader, n_imgs, device):
    res = np.zeros((n_imgs, 1), dtype = np.float32)
    for weights in models_w_arr:
        model.load_state_dict(weights) 
        model.eval()
        res_arr = []
        for inputs, _ in tqdm(loader):
            inputs = inputs.to(device)
            sz = inputs.shape
            inputs = inputs.reshape(-1,sz[2],sz[3],sz[4])
            # forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs) 
                outputs = outputs.reshape(sz[0], sz[1], 1).mean(1)
                res_arr.append(outputs.detach().cpu().numpy())
        res_arr = np.concatenate(res_arr, axis = 0)
        
        res += res_arr
    return res / len(models_w_arr)


# In[ ]:


def test_augment_feat(model, models_w_arr, loader, n_imgs, device):
    res = np.zeros((n_imgs, 2048), dtype = np.float32)
    for weights in models_w_arr:
        model.load_weights(weights) 
        model.eval()
        res_arr = []
        for inputs, _ in tqdm(loader):
            inputs = inputs.to(device)
            sz = inputs.shape
            inputs = inputs.reshape(-1,sz[2],sz[3],sz[4])
            # forward
            with torch.set_grad_enabled(False):
                outputs = model.extract_features(inputs) 
                outputs = outputs.reshape(sz[0], sz[1], -1).mean(1)
                res_arr.append(outputs.detach().cpu().numpy())
        res_arr = np.concatenate(res_arr, axis = 0)
        
        res += res_arr
    return res / len(models_w_arr)


# In[6]:


class WindDataset(Dataset):
    def __init__(self, imgs, gts, split_type, index, transform):
        if split_type == 'test':
            self.imgs = imgs   
        else:
            self.imgs = [imgs[idx] for idx in index]
            self.gts = [gts[idx] for idx in index]   
        
        self.split_type = split_type
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        #img[img == 255] = 0
        #img = Image.fromarray(img)
        img = self.transform(img)
        if self.split_type == 'test':
            return img, 0.0
        return img, self.gts[idx]


# In[7]:


class WindTestDataset(Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs 
        
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        arr = []

        for angle in range(0,179,45):
            rot_img = torchvision.transforms.functional.rotate(img, angle, False, False, None, None)
            rot_img = self.transform(rot_img).unsqueeze(0)
            arr.append(rot_img)

        for angle in range(-45,-179,-45):
            rot_img = torchvision.transforms.functional.rotate(img, angle, False, False, None, None)
            rot_img = self.transform(rot_img).unsqueeze(0)
            arr.append(rot_img)

        imgs = torch.cat(arr)
        return imgs, 0.0


# In[8]:


class ConvNet(nn.Module):
    def __init__(self, model):
        super(ConvNet, self).__init__()
        self.model = model
        
    def extract_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        #print(x.shape)
        
        x = self.model.avgpool(x).squeeze(-1).squeeze(-1)
        return x
    
    def load_weights(self, weights):
        self.model.load_state_dict(weights)


# In[10]:


# In[11]:


# In[12]:


#train_imgs = np.array(train_imgs).astype(np.uint8)


# In[13]:


#train_imgs.shape


# In[14]:


#train_imgs.mean(), train_imgs.std()


# In[15]:


#group_kfold = GroupShuffleSplit(n_splits=5, random_state = 4321)
group_kfold = GroupKFold(n_splits=5)


# In[16]:


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),
        transforms.RandomAffine(degrees = 45, scale = (0.9,1.1)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[ ]:


path = 'resnet50_models'


# In[17]:

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)
model_ft = model_ft.to(device)

# In[18]:

models_w_arr = []
for fold in range(5):
    for i in range(2):
        models_w_arr.append(torch.load(path + '/fold_%d/%d.pth'%(fold, i)))

# In[19]:


test_df = pd.read_csv('test_set_features.csv')


# In[20]:


test_imgs = []
for img_id in test_df['image_id']:
    img = Image.open('re-test-images/test/%s.jpg'%(img_id))
    test_imgs.append(img.copy())
    img.close()


# In[21]:


test_dataset = WindTestDataset(test_imgs, data_transforms['val'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128//8,shuffle=False, num_workers=8)


# In[23]:

test_pred = test_augment(model_ft, models_w_arr, test_loader, len(test_dataset), device)


# In[24]:


sub = pd.read_csv('submission_format.csv')
sub['image_id'] = test_df['image_id']
sub['wind_speed'] = np.round(test_pred[:,0]).astype(np.int64)
#sub.to_csv('res50_mae_adam_2cyc_group_5folds_aug_flip_rot_scale_tts_rot45.csv', index = False)


# In[29]:


# In[32]:


# In[33]:


test_df['pred'] = test_pred[:,0]
print(test_df.head())
test_df.to_csv(path+'/test.csv', index=False)


# In[ ]:

encapsulated_model_ft = ConvNet(model_ft)


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


test_features = test_augment_feat(encapsulated_model_ft, models_w_arr, test_loader, len(test_dataset), device)


# In[ ]:


np.save(path+'/test_features', test_features)

