#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""
Script to create training examples and TFRecord files for image-time-series tasks.
Process includes two stages:
    1) creation of numpy arrays containing specified number of images stored as byte strings
    2) creation of TFRecord files form numpy arrays where list of images is stored as BytesList
Process adheres requirement that "model cannot look in the future"
i.e. for any given current frame we stack ONLY PREVIOUS frames or copies of 1st frame as padding.

Resulting data may be used with TimeDistributed Conv2D models or Conv3D models.

Important: current frame is stored as LAST item in the list, i.e. when we have 24 frames current frame is `frames[23]`
"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./', help='Data directory')
parser.add_argument('--n_frames', type=int, default=24, help='Number of frames to stack.')
parser.add_argument('--step', type=int, default=1, help='Step between frames. Total frame depth is (n_frames * step)')
cmd = parser.parse_args()
print('Command line arguments:')
print(cmd)

import os
import glob
import shutil
import collections
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
print('tf:', tf.__version__)

n_frames = cmd.n_frames
step = cmd.step
data_dir = cmd.data_dir
prefix = os.path.join(data_dir, 'data-numpy-frames%03d-step%03d-lstm-' % (n_frames, step))
tfrec_dir = os.path.join(data_dir, 'data-tfrec-frames%03d-step%03d-lstm' % (n_frames, step))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Creating Numpy...')

train_df = pd.read_csv(os.path.join(data_dir, 'train_cv.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_cv.csv'))

# Make dirs for train/test numpy arrays
for d in ['train', 'test']:
    d = prefix + d
    if not os.path.exists(d):
        os.makedirs(d)

# Make dir for TFREcords
if not os.path.exists(tfrec_dir):
        os.makedirs(tfrec_dir)

# Add 'a' before 'train' in order to sort 'atrain' before 'test'
train_df['image_path'] = 'a' + train_df['image_path']

# To ease concat
test_df['fold_id'] = 0

all_df = pd.concat([train_df, test_df])
print(all_df.shape) # (114634, 11)

# Create ordered by time (i.e. by frame index) series of file names corresponding to frames
sorted_image_paths_df = all_df.groupby('storm_id')['image_path'].apply(sorted).reset_index(name='sorted_image_paths')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for storm_id, row in sorted_image_paths_df.iterrows():
    sorted_image_paths = np.array(row['sorted_image_paths'])
    for frame_id_current, path_current in enumerate(sorted_image_paths):
        #
        frame_ids = []
        for i in range(n_frames * step)[::-1]:
            delta = frame_id_current - i
            if delta < 0:
                delta = 0
            frame_ids.append(delta)
        #
        frame_ids = frame_ids[::-1][::step][:n_frames][::-1]
        #
        files_to_stack = sorted_image_paths[frame_ids]
        files_to_stack = [file.replace('atrain', 'train') for file in files_to_stack]
        #
        # Compliance check
        # print('----')
        # print(path_current)
        # print(frame_ids)
        # print(files_to_stack)
        #
        files_to_stack_bytes = []
        for file in files_to_stack:
            with open(os.path.join(data_dir, file), 'rb') as f:
                files_to_stack_bytes.append(f.read())
        np.save(prefix + path_current.replace('atrain', 'train').replace('.jpg', '.npy'), np.array(files_to_stack_bytes))
        #
    if storm_id % 50 == 0:
        print('Processed %d storms of 638' % storm_id)

print('Numpy DONE')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Creating TFRecord...')
  
class TFRecordProcessor(object):
    #
    def __init__(self, base_dir='./'):
        self.n_examples = 0
        self.base_dir = base_dir
    #
    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    #
    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    #
    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    #
    def _process_example(self, ind, X_id, X_name, Y, Y_cls):
        self.n_examples += 1
        feature = collections.OrderedDict()
        #
        feature['image_id'] = self._bytes_feature([X_id[ind].encode('utf-8')])
        feature['image']    = self._bytes_feature( list(np.load(os.path.join(self.base_dir, X_name[ind]))) )
        feature['label']    = self._int_feature(Y[ind])
        feature['label_le'] = self._int_feature(Y_cls[ind])
        #
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        #
        self._writer.write(example_proto.SerializeToString())
    #
    def write_tfrecords(self, X_id, X_name, Y, Y_cls, n_shards=1, file_out='train.tfrecord', print_each=1000):
        n_examples_per_shard = X_name.shape[0] // n_shards
        n_examples_remainder = X_name.shape[0] %  n_shards   
        self.n_examples = 0
        #
        for shard in range(n_shards):
            self._writer = tf.io.TFRecordWriter('%s-%05d-of-%05d' % (file_out, shard, n_shards))
            #
            start = shard * n_examples_per_shard
            if shard == (n_shards - 1):
                end = (shard + 1) * n_examples_per_shard + n_examples_remainder
            else:
                end = (shard + 1) * n_examples_per_shard
            #
            print('Shard %d of %d: (%d examples)' % (shard, n_shards, (end - start)))
            for i in range(start, end):
                self._process_example(i, X_id, X_name, Y, Y_cls)
                if not i % print_each:
                    print(i)
            #
            self._writer.close()
        #
        return self.n_examples

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Load data
train_df = pd.read_csv(os.path.join(data_dir, 'train_cv.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_cv.csv'))

# Create paths to numpy arrays
train_df['image_path'] = prefix + train_df['image_path']
test_df['image_path'] = prefix + test_df['image_path']
#
train_df['image_path'] = train_df['image_path'].str.replace('.jpg', '.npy')
test_df['image_path'] = test_df['image_path'].str.replace('.jpg', '.npy')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

tfrp = TFRecordProcessor()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for fold_id in range(len(train_df['fold_id'].unique())):
    print('Fold:', fold_id)
    n_written = tfrp.write_tfrecords(
        train_df[train_df['fold_id'] == fold_id]['image_id'].values,
        train_df[train_df['fold_id'] == fold_id]['image_path'].values,
        train_df[train_df['fold_id'] == fold_id]['wind_speed'].values,
        train_df[train_df['fold_id'] == fold_id]['wind_speed_le'].values,
        #
        n_shards=1, 
        file_out=os.path.join(tfrec_dir, 'fold.%d.tfrecord' % fold_id), 
        print_each=1000)

#----

n_written = tfrp.write_tfrecords(
    test_df['image_id'].values,
    test_df['image_path'].values,
    test_df['wind_speed'].values,
    test_df['wind_speed_le'].values,
    #
    n_shards=1, 
    file_out=os.path.join(tfrec_dir, 'test.tfrecord'),
    print_each=1000)


print('TFRecord DONE')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Removing Numpy...')

for d in ['train', 'test']:
    d = prefix + d
    shutil.rmtree(d, ignore_errors=True)

print('DONE')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


