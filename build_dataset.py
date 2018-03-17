"""Split the KITTI dataset into train/val/test and resize images to 404x404."""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

import utils

SIZE = 404

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/KITTI', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/404x404_KITTI', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, os.path.basename(filename)))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'training', 'image_2')
    train_gt_data_dir = os.path.join(args.data_dir, 'training', 'gt_image_2')
    test_data_dir = os.path.join(args.data_dir, 'testing', 'image_2')

    # Get the filenames in each directory (train and test)
    train_val_files = os.listdir(train_data_dir)
    train_val_gt_files= os.listdir(train_gt_data_dir)
    test_files = os.listdir(test_data_dir)

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    train_val_files.sort()
    random.shuffle(train_val_files)

    split = int(0.8 * len(train_val_files))
    train_files, val_files = train_val_files[:split], train_val_files[split:]

    train_gt_files = [utils.data_file_to_gt(train_file) for train_file in train_files]
    val_gt_files = [utils.data_file_to_gt(val_file) for val_file in val_files]

    sources = {'train': (train_data_dir, train_files + train_gt_files),
               'val': (train_data_dir, val_files + val_gt_files),
               'test': [test_files]}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    train_output_dir = os.path.join(args.output_dir, 'train')
    val_output_dir = os.path.join(args.output_dir, 'val')
    test_output_dir = os.path.join(args.output_dir, 'test')

    # Preprocess train, dev and test
    data_map = [(train_data_dir, train_files, train_output_dir),
                (train_gt_data_dir, train_gt_files, train_output_dir),
                (train_data_dir, val_files, val_output_dir),
                (train_gt_data_dir, val_gt_files, val_output_dir),
                (test_data_dir, test_files, test_output_dir)]

    for data_dir, data_files, output_dir in data_map:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir))
        for data_file in tqdm(data_files):
            from_path = os.path.join(data_dir, data_file)
            resize_and_save(from_path, output_dir, size=SIZE)

    print("Done building dataset")
