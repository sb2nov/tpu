# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf

import resnet_preprocessing
from tensorflow.contrib.data.python.ops import batching


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = resnet_preprocessing.preprocess_image(
        image_buffer=image_bytes, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


class ImageNetInput(object):
  def __init__(self, is_training,
               data_dir,
               num_cores=8,
               num_parallel_calls=64,
               use_transpose=False):
    pass

  def input_fn(self, params):
    batch_size = params['batch_size']
    dataset = tf.data.Dataset.range(1).repeat().map(
      lambda x: (tf.cast(tf.constant(np.zeros((224, 224, 3)).astype(np.float32), tf.float32), tf.bfloat16),
                 tf.constant(0, tf.int32)))

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(4)     # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()
    images = tf.transpose(images, [1, 2, 3, 0])
    return images, labels
