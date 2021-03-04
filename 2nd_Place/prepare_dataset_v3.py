import pandas as pd
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--step_size', help='step size between each time step', required=True, type=int)
args = parser.parse_args()

time_steps = 6
step_size = args.step_size
new_img_shape = (224,224)

train_df = pd.read_csv('training_set_features.csv')
train_df.set_index("image_id", inplace=True)

train_labels = pd.read_csv('training_set_labels.csv')
train_labels['wind_speed'] = train_labels['wind_speed'].astype(np.float32)
train_labels.set_index("image_id", inplace=True)

train_df = train_df.join(train_labels, on='image_id')
train_df['split'] = 'train'

test_df = pd.read_csv('test_set_features.csv')
test_df.set_index("image_id", inplace=True)
test_df['split'] = 'test'
test_df['wind_speed'] = -1

df = pd.concat([train_df, test_df], axis = 0)

test_imgs = []
test_imgs_ids = []

for _, storm_df in df.groupby('storm_id'):
    storm_df = storm_df.sort_values('relative_time')
    
    n = storm_df.shape[0]
    imgs_arr = np.zeros((n,) + new_img_shape, dtype=np.uint8)
    
    for i in range(n):
        img_id = storm_df.index[i]
        split = storm_df['split'].iloc[i]
        target = storm_df['wind_speed'].iloc[i]
        storm_id = storm_df['storm_id'].iloc[i]
        
        img = Image.open('re-%s-images/%s/%s.jpg'%(split, split, img_id))
        img = img.resize(new_img_shape, resample=Image.BILINEAR)
        imgs_arr[i] = np.array(img.copy()).astype(np.uint8)
        imgs_arr[i][imgs_arr[i] == 255] = 0
        img.close()
        
        if split == 'test':
            idx_arr = [i]
            for j in range(i-step_size, i-time_steps*step_size, -1*step_size):
                if j < 0:
                    idx_arr.append(idx_arr[-1])
                else:
                    idx_arr.append(j)
            
            img_arr = [imgs_arr[j] for j in idx_arr]
            img_arr = [np.array(img_arr[j:j+3]).transpose(1,2,0) for j in range(0, time_steps, 3)]
            
            sample = np.concatenate(img_arr, 2)

            test_imgs.append(sample)
            test_imgs_ids.append(img_id)


test_imgs = np.array(test_imgs)
test_imgs_ids = np.array(test_imgs_ids)

print(test_imgs.shape)

np.save('6ts_%dspace_test_imgs_%d'%(step_size, new_img_shape[0]), test_imgs)
np.save('6ts_%dspace_test_imgs_ids'%(step_size), test_imgs_ids)
