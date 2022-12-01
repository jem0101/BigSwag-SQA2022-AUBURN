"""
Copyright (C) 2019  Syed Hasibur Rahman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# region : Imports
import abc
import tensorflow as tf
import tensorflow_datasets as tfds

from easydict import EasyDict
from synapse.config import *
from synapse.utilities import check_initialize
# end region : Imports

# region : Enum


class AudioDatasets(Enum):
    groove = 1001
    nsynth = 1002


class ImageDatasets(Enum):
    abstract_reasoning = 2001
    aflw2k3d = 2002
    bigearthnet = 2003
    binarized_mnist = 2004
    binary_alpha_digits = 2005
    caltech101 = 2006
    caltech_birds2010 = 2007
    caltech_birds2011 = 2008
    cats_vs_dogs = 2009
    celeb_a = 2010
    celeb_a_hq = 2011
    cifar10 = 2012
    cifar100 = 2013
    cifar10_corrupted = 2014
    clevr = 2015
    coco = 2016
    coco2014 = 2017
    coil100 = 2018
    colorectal_histology = 2019
    colorectal_histology_large = 2020
    curated_breast_imaging_ddsm = 2021
    cycle_gan = 2022
    deep_weeds = 2023
    diabetic_retinopathy_detection = 2024
    downsampled_imagenet = 2025
    dsprites = 2026
    dtd = 2027
    emnist = 2028
    eurosat = 2029
    fashion_mnist = 2030
    food101 = 2031
    horses_or_humans = 2032
    image_label_folder = 2033
    imagenet2012 = 2034
    imagenet2012_corrupted = 2035
    kitti = 2036
    kmnist = 2037
    lfw = 2038
    lsun = 2039
    mnist = 2040
    mnist_corrupted = 2041
    omniglot = 2042
    open_images_v4 = 2043
    oxford_flowers102 = 2044
    oxford_iiit_pet = 2045
    patch_camelyon = 2046
    pet_finder = 2047
    quickdraw_bitmap = 2048
    resisc45 = 2049
    rock_paper_scissors = 2050
    scene_parse150 = 2051
    shapes3d = 2052
    smallnorb = 2053
    so2sat = 2054
    stanford_dogs = 2055
    stanford_online_products = 2056
    sun397 = 2057
    svhn_cropped = 2058
    tf_flowers = 2059
    uc_merced = 2060
    visual_domain_decathlon = 2061
    voc2007 = 2062


class StructuredDatasets(Enum):
    amazon_us_reviews = 3001
    higgs = 3002
    iris = 3003
    rock_you = 3004
    titanic = 3005


class TextDatasets(Enum):
    cnn_dailymail = 4001
    definite_pronoun_resolution = 4002
    gap = 4003
    glue = 4004
    imdb_reviews = 4005
    lm1b = 4006
    multi_nli = 4007
    snli = 4008
    squad = 4009
    super_glue = 4010
    trivia_qa = 4011
    wikipedia = 4012
    xnli = 4013


class TranslateDatasets(Enum):
    flores = 5001
    para_crawl = 5002
    ted_hrlr_translate = 5003
    ted_multi_translate = 5004
    wmt14_translate = 5005
    wmt15_translate = 5006
    wmt16_translate = 5007
    wmt17_translate = 5008
    wmt18_translate = 5009
    wmt19_translate = 5010
    wmt_t2t_translate = 5011


class VideoDatasets(Enum):
    bair_robot_pushing_small = 6001
    moving_mnist = 6002
    starcraft_video = 6003
    ucf101 = 6004


# endregion : Enum


class DatasetInfo:
    total_count = 0
    total_epoch = 0
    num_max_boxes = 0


class BaseDataset:
    def __init__(self, name, download_dir, auto_initialize):
        self.name = name
        self._classes = None
        self.download_dir = download_dir

        self.subset = None
        self.ds_builder = None
        self.datasets = {}
        self.data_dir = None
        self.batch_size = {}
        self._initialized = False
        self._info = DatasetInfo()

        if auto_initialize:
            self.initialize()

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    @check_initialize
    @abc.abstractmethod
    def __call__(self, subset):
        raise NotImplementedError

    @check_initialize
    @property
    def classes(self):
        return self._classes

    @property
    def info(self):
        return self._info

    @property
    def is_initialized(self):
        return self._initialized

    @check_initialize
    @property
    def total_data_count(self):
        raise NotImplementedError

    @check_initialize
    @property
    def data_count(self, subset):
        raise NotImplementedError


class PublicDataset(BaseDataset):
    """
    Interface class to generate and provide all available public dataset.
    """

    def __init__(self,
                 name=ImageDatasets.mnist,
                 download_dir=None,
                 auto_initialize=True):
        super(PublicDataset, self).__init__(name,
                                            download_dir,
                                            auto_initialize)
        self._class_label = None
        self._num_classes = None

    def initialize(self):
        print("Download and Prepare the data .........", self.name)
        if not self._initialized:
            ds_name = self.name.name if isinstance(self.name, Enum) else str(self.name)
            ds_builder = tfds.builder(ds_name.lower())
            ds_builder.download_and_prepare(download_dir=self.download_dir)
            self.data_dir = ds_builder.data_dir
            self.ds_builder = ds_builder
            self._info = self.ds_builder.info
            self._classes = self.ds_builder.info.features['label']
            self._initialized = self.ds_builder.info.initialized
        return self._initialized

    @staticmethod
    def normalize_img(image, label):
        return tf.cast(image, dtype=tf.float32) / 255., label

    @staticmethod
    def channel_first(image, label):
        return tf.transpose(image, [2, 0, 1]), label

    @check_initialize
    def __call__(self, subset,
                 data_format=DataFormat.ChannelsLast,
                 shuffle=True,
                 batch=1,
                 prefetch=True,
                 repeat=0,
                 in_memory=True,
                 normalize_img=True):
        assert subset in ['train', 'test'], "Invalid subset : %s. Must be from : ['train', 'test']" % subset

        ds_subset = self.datasets.get(subset, None)

        if not ds_subset:
            ds_subset = self.ds_builder.as_dataset(split=subset, in_memory=in_memory, as_supervised=True)
            if normalize_img:
                ds_subset = ds_subset.map(self.normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if data_format == DataFormat.ChannelsFirst:
                ds_subset = ds_subset.map(self.channel_first, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_subset = ds_subset.cache()
            if shuffle:
                num_examples = self.info.splits[subset].num_examples
                ds_subset = ds_subset.shuffle(num_examples)
            ds_subset = ds_subset.batch(batch)
            if repeat:
                ds_subset = ds_subset.repeat(repeat)
            ds_subset = ds_subset.prefetch(tf.data.experimental.AUTOTUNE)

            self.datasets[subset] = ds_subset
            self.batch_size[subset] = batch

        return ds_subset

    @check_initialize
    @property
    def classes(self):
        return self._classes.names

    @property
    def num_classes(self):
        return self._classes.num_classes

    @property
    def total_data_count(self):
        return self.info.splits.total_num_examples if self.is_initialized else None

    def data_count(self, subset):
        return self.info.splits[subset].num_examples if self.is_initialized else None
