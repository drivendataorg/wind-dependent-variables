import os

import numpy as np
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset


TRAIN_FEATURES = "training_set_features.csv"
TEST_FEATURES = "test_set_features.csv"
TRAIN_LABELS = "training_set_labels.csv"

DATASET_MEAN = 69.299
DATASET_STD = 59.048


def get_storms_df(data_dir, include_path=True):
    """Return DataFrame with dataset

    :param data_dir: directory of project data
    :param include_path: whether to include path to image
    """
    # Read train set data
    train_df = pd.read_csv(os.path.join(data_dir, TRAIN_FEATURES))
    train_df.set_index('image_id', inplace=True, drop=False)

    # Read train labels
    train_labels = pd.read_csv(os.path.join(data_dir, TRAIN_LABELS))
    train_labels.set_index('image_id', inplace=True, drop=True)
    train_df = pd.merge(train_df, train_labels, how='left', left_index=True, right_index=True)
    train_df['train'] = True
    train_df['test'] = False

    # Read test set data
    test_df = pd.read_csv(os.path.join(data_dir, TEST_FEATURES))
    test_df.set_index('image_id', inplace=True, drop=False)
    test_df['wind_speed'] = None
    test_df['train'] = False
    test_df['test'] = True

    # Concatenate datasets
    dset_df = pd.concat([train_df, test_df])
    dset_df.reset_index(drop=True, inplace=True)
    dset_df.sort_values(['image_id'], inplace=True)
    dset_df.reset_index(drop=True, inplace=True)

    # Add storm group
    train_storms = dset_df.storm_id[dset_df.train].unique()
    test_storms = dset_df.storm_id[dset_df.test].unique()
    only_train = [s1 for s1 in train_storms if s1 not in test_storms]
    only_test = [s1 for s1 in test_storms if s1 not in train_storms]

    def storm_group(storm_id):
        if storm_id in only_train:
            return 'train'
        elif storm_id in only_test:
            return 'test'
        return 'common'

    dset_df['group'] = dset_df.apply(lambda x: storm_group(x.storm_id), axis=1)

    # Add delta_time
    arr_v = dset_df.relative_time.values
    arr_i = dset_df.storm_id.values
    dset_df['delta_time'] = np.concatenate([[0,], (arr_v[1:] - arr_v[:-1])*np.equal(arr_i[1:], arr_i[:-1])])

    # Add storm duration in hours
    dset_df['storm_duration'] = (dset_df.relative_time / 1800).round() / 2

    # Add wind_speed_bin
    dset_df['wind_speed_bin'] = None
    clipped = np.clip(dset_df.wind_speed[dset_df.train].values, 0, 150)
    hist = np.histogram(clipped, 15)[1][1:]
    dset_df.loc[dset_df.train, 'wind_speed_bin'] = np.array([np.where(s1 <= hist)[0][0] for s1 in clipped])

    # Add path
    if include_path:
        def add_path(irow):
            return os.path.join(data_dir, 'train' if irow.train else 'test', irow.image_id + '.jpg')
        dset_df['image_path'] = dset_df.apply(add_path, axis=1)

    return dset_df


class StormsDataset(Dataset):
    """
    Pytorch Dataset class that returns a single RBG image where all channels have the same raw 1-channel image"
    """

    def __init__(self, dataframe, annot_column='wind_speed', image_mode='RGB', max_block_size=None):
        """Initialize class.

        Take a dataframe with images path in *path*

        :param dataframe: DataFrame.
            DataFrame with dataset
        :param annot_column: String, default 'wind_speed'.
            DataFrame column's name with the annotations.
        :param image_mode: String, default 'RGB'.
            Image mode for outputs, RGB or L (grey).
        :param max_block_size: Int or None.
            Maximum size of blocks of sequential images with similar annotation. None for not use blocks.
        """

        self.df = dataframe.copy()
        # Ensure dataframe are correctly sorted
        self.df.sort_values('image_id', ascending=True)
        self.annot_column = annot_column
        assert image_mode in ['RGB', 'L']
        self.image_mode = image_mode

        self.max_block_size = max_block_size
        if self.max_block_size is None:
            # Normal mode
            self.blocks = False
        else:
            # Blocks mode
            self.blocks = True
            self.blocks_grs = self._get_blocks()
            self.iter_df = pd.DataFrame(self.blocks_grs.count().index.values, columns=['gr_index'])

    def _get_blocks(self):
        """
        Function that groups data for each storm in blocks of specific size.
        For each storm id, blocks are built as follow:
            - Create first block with id=0 and add first image.
            - Next image will belong to the same block if the difference between max and min wind_speed
            within the block would be less or equal to `max_block_size`. If not, new block is created
            with id += 1.
            -
        Return a pandas DataFrameGroupBy object.
        """
        result = []
        for storm_id, gr in self.df[self.df.train].groupby('storm_id'):
            tmp_df = gr.sort_values('relative_time', ascending=True)
            block_id, block_min, block_max = 0, None, None
            for image_idx, image_id, wind_speed in zip(tmp_df.index, tmp_df.image_id, tmp_df.wind_speed):

                if block_min is None or wind_speed < block_min:
                    new_block_min = wind_speed
                if block_max is None or wind_speed > block_max:
                    new_block_max = wind_speed

                new_block_size = new_block_max - new_block_min

                if new_block_size > self.max_block_size:
                    block_id += 1
                    new_block_min = wind_speed
                    new_block_max = wind_speed

                block_min = new_block_min
                block_max = new_block_max
                result.append([image_idx, storm_id, image_id, wind_speed, block_id])
        return pd.DataFrame(result, columns=['image_idx', 'storm_id', 'image_id', 'wind_speed', 'block_id']). \
            groupby(['storm_id', 'block_id'])

    def get_iter_df(self):
        if self.blocks:
            return self.iter_df
        return self.df

    def __len__(self):
        return len(self.get_iter_df())

    def __getitem__(self, ix):
        irow = self.get_iter_df().iloc[ix]
        return self.read(irow)

    def read(self, irow, features=None):
        if self.blocks:
            # Randomly sample images from a specific block
            irow = self.df.loc[np.random.choice(self.blocks_grs.get_group(irow.gr_index).image_idx.values)]

        im = Image.open(irow.image_path).convert(self.image_mode)
        annot = np.array(irow[self.annot_column], np.float32)

        if features is None:
            # return PIL_image, label, info dictionary
            return im, annot, {}

        if not isinstance(features, list):
            features = [features, ]

        exf = []
        if 'ocean' in features:
            exf.append((irow.ocean - 1.5) / 0.5)  # Standardization
        if 'delta_time':
            exf.append(((round(irow.delta_time / 1800, 0) / 2) - 0.735) / 1.769)  # Standardization
        if 'storm_duration':
            exf.append(irow.storm_duration)

        return [im, np.array(exf, np.float32)], annot, {}


class StormsDatasetRGBSequence(Dataset):
    """
    Pytorch Dataset class that returns a single RBG image composed by three different 1-channel images"
    """

    def __init__(self, dataframe, gap, annot_column='wind_speed', max_block_size=None):
        """Initialize class.

        Take a dataframe with images path in *path*

        :param dataframe: DataFrame.
            DataFrame with dataset
        :param gap: Float.
            Gap, in hours, for consecutive images the images
        :param annot_column: String, default 'wind_speed'.
            DataFrame column's name with the annotations.
        :param max_block_size: Int or None.
            Maximum size of blocks of sequential images with similar annotation. None for not use blocks.
        """
        self.df = dataframe.copy()
        self.gap = gap  # Gap, in hours, between the images
        self.annot_column = annot_column
        # Ensure dataframe are correctly sorted
        self.df.sort_values('image_id', ascending=True)
        # Storms
        self.storm_df = self.df.copy()
        self.storm_df.set_index(['storm_id', 'storm_duration'], inplace=True, drop=False)

        self.max_block_size = max_block_size
        if self.max_block_size is None:
            # Normal mode
            self.blocks = False
        else:
            # Blocks mode
            self.blocks = True
            self.blocks_grs = self._get_blocks()
            self.iter_df = pd.DataFrame(self.blocks_grs.count().index.values, columns=['gr_index'])

    def _get_blocks(self):
        """
        Function that groups data for each storm in blocks of specific size.
        For each storm id, blocks are built as follow:
            - Create first block with id=0 and add first image.
            - Next image will belong to the same block if the difference between max and min wind_speed
            within the block would be less or equal to `max_block_size`. If not, new block is created
            with id += 1.
            -
        Return a pandas DataFrameGroupBy object.
        """
        result = []
        for storm_id, gr in self.df[self.df.train].groupby('storm_id'):
            tmp_df = gr.sort_values('relative_time', ascending=True)
            block_id, block_min, block_max = 0, None, None
            for image_idx, image_id, wind_speed in zip(tmp_df.index, tmp_df.image_id, tmp_df.wind_speed):

                if block_min is None or wind_speed < block_min:
                    new_block_min = wind_speed
                if block_max is None or wind_speed > block_max:
                    new_block_max = wind_speed

                new_block_size = new_block_max - new_block_min

                if new_block_size > self.max_block_size:
                    block_id += 1
                    new_block_min = wind_speed
                    new_block_max = wind_speed

                block_min = new_block_min
                block_max = new_block_max
                result.append([image_idx, storm_id, image_id, wind_speed, block_id])
        return pd.DataFrame(result, columns=['image_idx', 'storm_id', 'image_id', 'wind_speed', 'block_id']). \
            groupby(['storm_id', 'block_id'])

    def get_iter_df(self):
        if self.blocks:
            return self.iter_df
        return self.df

    def __len__(self):
        return len(self.get_iter_df())

    def __getitem__(self, ix):
        irow = self.get_iter_df().iloc[ix]
        return self.read(irow)

    def read(self, irow):
        if self.blocks:
            # Randomly sample images from a specific block
            irow = self.df.loc[np.random.choice(self.blocks_grs.get_group(irow.gr_index).image_idx.values)]

        im_R = Image.open(irow.image_path).convert('L')
        try:
            # Note the negative sign to pick previous images in time
            path = self.storm_df.loc[[(irow.storm_id, irow.storm_duration - self.gap)]].iloc[0].image_path
            im_G = Image.open(path).convert('L')
        except KeyError:
            im_G = im_R
        try:
            # Note the negative sign to pick previous images in time
            path = self.storm_df.loc[[(irow.storm_id, irow.storm_duration - 2 * self.gap)]].iloc[0].image_path
            im_B = Image.open(path).convert('L')
        except KeyError:
            im_B = im_G

        im = Image.merge('RGB', (im_R, im_G, im_B))
        annot = np.array(irow[self.annot_column], np.float32)

        # return PIL_image, label, info dictionary
        return im, annot, {}


class StormsDatasetSequence(Dataset):
    """
    Pytorch Dataset class that returns a sequence of 1-channel images"
    """

    def __init__(self, dataframe, nb_imgs, gap, annot_column='wind_speed', missing='previous', max_block_size=None,
                 return_all_labels=False, reduce_all_labels=None):
        """Initialize class.

        Take a dataframe with images path in *path*

        :param dataframe: DataFrame.
            DataFrame with dataset
        :param nb_imgs: Int.
            Number of images or length of sequence
        :param gap: Float.
            Gap, in hours, for consecutive images the images.
            If -1, pick previous image in the data set independently of the timestamp
        :param annot_column: String, default 'wind_speed'.
            DataFrame column's name with the annotations.
        :param missing: String, default 'previous'.
            What to do if a image is missing or doesn't exist.
            'previous' to use the previous image in the sequence. For the first image in a storm, the sequence will
            have this image repeated `nb_imgs` times.
            'black' to insert a black image.
        :param max_block_size: Int or None.
            Maximum size of blocks of sequential images with similar annotation. None for not use blocks.
        :param return_all_labels: Boolean, default False.
            Whether to return all labels for each image in the sequence or just the last one in time (first in the
            sequence)
        :param return_all_labels: Int, default None.
            Reduce labels of sequence of images, e.g. return_all_labels=3: to train models as sequence of RGB images,
            the images are grouped in three, and the label for each group or RGB image will be the latest of the group.
        """
        self.df = dataframe.copy()
        self.nb_imgs = nb_imgs
        self.gap = gap
        assert missing in ['previous', 'black']
        self.missing = missing
        self.annot_column = annot_column
        self.return_all_labels = return_all_labels
        self.reduce_all_labels = reduce_all_labels
        # Ensure dataframe are correctly sorted
        self.df.sort_values('image_id', ascending=True)
        self.df.reset_index(inplace=True, drop=True)
        # Storms
        self.storm_df = self.df.copy()
        self.storm_df.set_index(['storm_id', 'storm_duration'], inplace=True, drop=False)

        self.max_block_size = max_block_size
        if self.max_block_size is None:
            # Normal mode
            self.blocks = False
        else:
            # Blocks mode
            self.blocks = True
            self.blocks_grs = self._get_blocks()
            self.iter_df = pd.DataFrame(self.blocks_grs.count().index.values, columns=['gr_index'])

    def _get_blocks(self):
        """
        Function that groups data for each storm in blocks of specific size.
        For each storm id, blocks are built as follow:
            - Create first block with id=0 and add first image.
            - Next image will belong to the same block if the difference between max and min wind_speed
            within the block would be less or equal to `max_block_size`. If not, new block is created
            with id += 1.
            -
        Return a pandas DataFrameGroupBy object.
        """
        result = []
        for storm_id, gr in self.df[self.df.train].groupby('storm_id'):
            tmp_df = gr.sort_values('relative_time', ascending=True)
            block_id, block_min, block_max = 0, None, None
            for image_idx, image_id, wind_speed in zip(tmp_df.index, tmp_df.image_id, tmp_df.wind_speed):

                if block_min is None or wind_speed < block_min:
                    new_block_min = wind_speed
                if block_max is None or wind_speed > block_max:
                    new_block_max = wind_speed

                new_block_size = new_block_max - new_block_min

                if new_block_size > self.max_block_size:
                    block_id += 1
                    new_block_min = wind_speed
                    new_block_max = wind_speed

                block_min = new_block_min
                block_max = new_block_max
                result.append([image_idx, storm_id, image_id, wind_speed, block_id])
        return pd.DataFrame(result, columns=['image_idx', 'storm_id', 'image_id', 'wind_speed', 'block_id']). \
            groupby(['storm_id', 'block_id'])

    def get_iter_df(self):
        if self.blocks:
            return self.iter_df
        return self.df

    def __len__(self):
        return len(self.get_iter_df())

    def __getitem__(self, ix):
        irow = self.get_iter_df().iloc[ix]
        return self.read(irow)

    def read(self, irow):
        if self.blocks:
            irow = self.df.loc[np.random.choice(self.blocks_grs.get_group(irow.gr_index).image_idx.values)]
            ix = irow.name
            strom_id = irow.storm_id
        else:
            ix = self.df[self.df.image_id == irow.image_id].iloc[0].name
            strom_id = irow.storm_id

        img = []
        annots = []


        img.append(np.array(Image.open(irow.image_path).convert('L')))
        annots.append(irow[self.annot_column])
        for i_im in range(1, self.nb_imgs):
            try:
                if self.gap == -1:
                    # Get just previous image
                    prev_irow = self.df.iloc[ix-1]
                    if prev_irow.storm_id == strom_id:
                        # Same storm
                        img.append(np.array(Image.open(prev_irow.image_path).convert('L')))
                        annots.append(prev_irow[self.annot_column])
                        # Note the negative sign to pick previous images in time
                        ix = ix-1
                    else:
                        # Different storm
                        if self.missing == 'previous':
                            img.append(img[-1])
                            annots.append(annots[-1])
                        else:
                            img.append(np.zeros_like(img[-1]))
                            annots.append(0)

                else:
                    # Note the negative sign to pick previous images in time
                    prev_irow = self.storm_df.loc[[(irow.storm_id, irow.storm_duration - self.gap * i_im)]].iloc[0]
                    path = prev_irow.image_path
                    img.append(np.array(Image.open(path).convert('L')))
                    annots.append(prev_irow[self.annot_column])
            except KeyError:
                if self.missing == 'previous':
                    img.append(img[-1])
                    annots.append(annots[-1])
                else:
                    img.append(np.zeros_like(img[-1]))
                    annots.append(0)
        img = np.stack(img)
        img = np.transpose(img, (1, 2, 0))
        annots = np.array(annots, np.float32)

        annot = np.array(irow[self.annot_column], np.float32)
        if self.reduce_all_labels is not None:
            annots = annots[::self.reduce_all_labels]

        # return NUMPY image, label, info dictionary
        if self.return_all_labels:
            return img, annots[np.newaxis, ...], {}
        else:
            return img, annot, {}

    def generate_gif(self, img):
        img = np.transpose(img, (2, 0, 1))
        imgs = [Image.fromarray(s1) for s1 in img]
        imgs[0].save('sample.gif', format='GIF', append_images=imgs[1:], save_all=True, duration=500, loop=0)
        return './sample.gif'
