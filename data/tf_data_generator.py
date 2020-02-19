import numpy as np
import cv2
import os
from tensorflow.python.keras.utils import to_categorical
import tensorflow as tf
#use https://www.tensorflow.org/tutorials/load_data/images
from ..parameters import *
import pathlib
import re

@tf.function()
def _parse_function(example_proto):


    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.VarLenFeature(tf.string),
    }
    for i in range(n):
        name = "mask_raw_" + str(i)
        features[name] = tf.io.VarLenFeature(tf.string)

    parsed_features = tf.io.parse_single_example(example_proto, features)
    height = parsed_features['height']
    width = parsed_features['width']
    img_raw_tensor = tf.squeeze(tf.sparse.to_dense(parsed_features['image_raw']))
    img_tensor = tf.dtypes.cast(tf.reshape( tf.io.decode_jpeg(img_raw_tensor), (height, width, -1)), tf.float32)
    annotation_list = []
    for i in range(n):
        decoded_annotation = tf.io.decode_png(tf.squeeze(tf.sparse.to_dense(parsed_features['mask_raw_' + str(i)])))
        binarized_annotation = tf.greater(decoded_annotation, 125)
        binarized_annotation= tf.dtypes.cast(binarized_annotation, tf.float32)
        annotation_list.append(tf.squeeze(tf.reshape(binarized_annotation, (height//2, width//2, -1))) )
    annotation_list = tf.transpose(annotation_list, [1,2,0 ])
    return img_tensor, annotation_list

# filename_queue = ["dataset_chunk_0.tfrecord"]
#
# dataset = tf.data.TFRecordDataset(filename_queue)
#
# dataset = dataset.map(_parse_function)
#
# # Shuffle the dataset
# # dataset = dataset.shuffle(buffer_size=4)
# dataset.cache()
# dataset = dataset.batch(batch_size)
#
# # Repeat the input indefinitly
# dataset = dataset.repeat()
#
# # Get batch X and y
# X = dataset.take(1)

# for rec in X :
#     print(rec[0].shape)
#     img = tf.squeeze(rec[0]).numpy()
#     cv2.imwrite("tensorimg.jpg", img)
#     # print("\n ")
#     print(rec[1].shape)
#     annotation_1 = tf.squeeze(rec[1]).numpy()[:, :, 0]
#     cv2.imwrite("tensorannotation.png", annotation_1*255)
#     print(tf.math.reduce_max(rec[1]))
#     print("")

def prepare_for_training(ds,batch_size,  cache=False, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(batch_size)
  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

def get_dataset(batch_size):
    filename_queue = ["dataset_chunk_0.tfrecord"]

    dataset = tf.data.TFRecordDataset(filename_queue)

    dataset = dataset.map(_parse_function)

    val_dataset = prepare_for_training(dataset.take(val_size), batch_size)
    train_dataset = prepare_for_training(dataset.skip(val_size), batch_size)

    return train_dataset, val_dataset


def get_dataset2():

    dataset  = tf.data.Dataset.from_tensor_slices((np.random.uniform(size=(1, 1000, 1000, 3)), np.random.uniform(size=(1, 500, 500, 6))))

    dataset = dataset.shuffle(4)
    dataset = dataset.repeat()
    dataset = dataset.batch(2)
    return dataset

# # Get batch X and y
# dataset, _ = get_dataset(2)
# X = dataset.take(12)
#
# for rec in X :
#     print(rec[0].shape)
#     # print("\n ")
#     print(rec[1].shape)
#     print("")

# dataset = get_dataset()
# test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
# test_set = tfds.load(dataset, split=test_split, as_supervised=True)
#
# print(0)