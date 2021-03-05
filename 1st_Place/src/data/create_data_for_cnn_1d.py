#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./', help='Data directory')
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

data_dir = cmd.data_dir
prefix = os.path.join(data_dir, './')
tfrec_dir = os.path.join(data_dir, 'data-tfrec-1d')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Make dir for TFRecord
if not os.path.exists(tfrec_dir):
    os.makedirs(tfrec_dir)

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
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #
    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    #
    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    #
    def _process_example(self, ind, X_id, X_name, Y, Y_cls):
        self.n_examples += 1
        feature = collections.OrderedDict()
        #
        feature['image_id'] = self._bytes_feature(X_id[ind].encode('utf-8'))
        feature['image']    = self._bytes_feature(tf.io.read_file(os.path.join(self.base_dir, X_name[ind])))
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



