import utils

import argparse
import glob
import os
import fcn8_vgg
from PIL import Image

import tensorflow as tf


def get_dataset(data_dir):
    data_train_dir = os.path.join(data_dir, 'train')
    train_files = glob.glob(os.path.join(data_train_dir, 'um_[0-9]*.png'))

    train_images = []
    train_gt_images = []

    for train_file in train_files:
        train_file_path = os.path.join(data_train_dir, train_file)
        train_gt_file = utils.data_file_to_gt(train_file)
        train_gt_path = os.path.join(data_train_dir, train_gt_file)

        train_images.append(Image.open(train_file))
        train_gt_images.append(Image.open(train_gt_file))

    train_images = np.array(train_images)
    train_gt_images = np.array(train_gt_images)

    return tf.data.Dataset.from_tensor_slices((train_images, train_gt_images))


def train(data_dir):
    dataset = get_dataset(data_dir)
    vgg_fcn = fcn8_vgg.FCN8VGG()
    init = tf.global_variable_initializer()
    feed_dict = {}

    with tf.name_scope("vgg_fcn"):
            vgg_fcn.build(images)

    with tf.Session() as sess:
        sess.run(init)
        sess.run([vgg_fcn.pred_up], feed_dict=feed_dict)



# TODO: pass in model and hyperparams
# TODO: add cli

if __name__ == "__main__":
    print(get_dataset())
