#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./', help='Data directory')
cmd = parser.parse_args()
print('Command line arguments:')
print(cmd)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

data_dir = cmd.data_dir

#------------------------------------------------------------------------------
# Load
#------------------------------------------------------------------------------

print('Load csv')

# Load
train_df = pd.read_csv(os.path.join(data_dir, 'training_set_labels.csv'))
train_features_df = pd.read_csv(os.path.join(data_dir, 'training_set_features.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'submission_format.csv'))
test_features_df = pd.read_csv(os.path.join(data_dir, 'test_set_features.csv'))

# Merge
train_df = pd.merge(train_df, train_features_df, on='image_id', how='left')
test_df  = pd.merge(test_df,  test_features_df,  on='image_id', how='left')

#------------------------------------------------------------------------------
# Create features
#------------------------------------------------------------------------------

print('Create features')

# Prepare relative file paths to be able uniformly read files during TFRecord creation
train_df['image_path'] = 'train/' + train_df['image_id'] + '.jpg'
test_df['image_path'] = 'test/' + test_df['image_id'] + '.jpg'

# Create frame_id
train_df[['storm_id_copy', 'frame_id']] = train_df['image_id'].str.split('_', expand=True)
train_df = train_df.drop('storm_id_copy', axis=1)
train_df['frame_id'] = train_df['frame_id'].astype(np.int32)
#
test_df[['storm_id_copy', 'frame_id']] = test_df['image_id'].str.split('_', expand=True)
test_df = test_df.drop('storm_id_copy', axis=1)
test_df['frame_id'] = test_df['frame_id'].astype(np.int32)

# Label encoded storm_id
le_si = LabelEncoder()
le_si = le_si.fit(list(train_df['storm_id']) + list(test_df['storm_id']))
train_df['storm_id_le'] = le_si.transform(train_df['storm_id'])
test_df['storm_id_le'] = le_si.transform(test_df['storm_id'])
# Check if consequtive and starts from 0
lst = list(train_df['storm_id_le']) + list(test_df['storm_id_le'])
assert list(np.sort(np.unique(lst))) == list(range(0, max(lst)+1)), 'Non-consequtive, or starts not from 0'

# Label encoded wind_speed (to use as class label)
le_ws = LabelEncoder()
le_ws = le_ws.fit(list(train_df['wind_speed']))
train_df['wind_speed_le'] = le_ws.transform(train_df['wind_speed'])
# Check if consequtive and starts from 0
lst = list(train_df['wind_speed_le'])
assert list(np.sort(np.unique(lst))) == list(range(0, max(lst)+1)), 'Non-consequtive, or starts not from 0'

# Template column for fold_id
train_df['fold_id'] = 0

# Fake label for test (just for compatibility)
test_df['wind_speed'] = 0

# Fake label for test (label-encoded for classification) (just for compatibility)
test_df['wind_speed_le'] = 0

#------------------------------------------------------------------------------
# CV
#------------------------------------------------------------------------------

print('Create split')

# Group split
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

for fold_id, (train_index, val_index) in enumerate(gkf.split(train_df, groups=train_df['storm_id'].values)):
    train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id

assert len(train_df['fold_id'].unique()) == n_splits, 'Inconsistent number of splits'

for i in range(n_splits):
    assert train_df[train_df['fold_id'] == i]['storm_id'].isin(train_df[train_df['fold_id'] != i]['storm_id']).sum() == 0, 'Groups are intersected'

# Shuffle
train_df = train_df.sample(frac=1.0, random_state=33)

# Save
train_df.to_csv(os.path.join(data_dir, 'train_cv.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'test_cv.csv'), index=False)

#------------------------------------------------------------------------------
# Create mapping for classification label
#------------------------------------------------------------------------------

unique_labels_sorted = np.sort(train_df['wind_speed'].unique())
np.save(os.path.join(data_dir, 'unique_labels_sorted.npy'), unique_labels_sorted)

print('DONE')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


