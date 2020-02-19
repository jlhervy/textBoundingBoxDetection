# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
import numpy as np
import cv2
import tensorflow as tf
from ..parameters import n
import os

# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_image_annotation_pairs_to_tfrecord(chunk, tfrecords_filename):
    """Writes given image/annotation pairs to the tfrecords file.
    The function reads each image/annotation pair given filenames
    of image and respective annotation and writes it to the tfrecord
    file.
    Parameters
    ----------
    filename_pairs : array of tuples (img_filepath, annotation_filepath)
        Array of tuples of image/annotation filenames
    tfrecords_filename : string
        Tfrecords filename to write the image/annotation pairs
    """
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    for path in chunk:
        basename = os.path.split(path)[1]
        img_path = os.path.join(path, basename + ".jpg")
        img = cv2.imread(img_path)

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]
        img_raw = open(img_path, 'rb').read()

        features_object = {
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw)
            }


        for i in range(n):
            name = "mask_raw_"+ str(i)
            features_object[name] = _bytes_feature(open(os.path.join(path, "mask_" + str(i) + ".png"), 'rb').read())

        example = tf.train.Example(features=tf.train.Features(feature=features_object))

        writer.write(example.SerializeToString())

    writer.close()



