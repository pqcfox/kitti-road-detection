"""Preprocesses the KITTI dataset and split into train/val/test."""

import argparse
import glob
import os
import random
import shutil
import utils
from tqdm import tqdm

SPLIT = 0.8
SEED = 230

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/KITTI', help="Directory with the KITTI dataset")
parser.add_argument('--output_dir', default='data/preproc_KITTI', help="Where to write the new data")

if __name__ == "__main__":
    args = parser.parse_args()

    train_data_dir = os.path.join(args.data_dir, 'training', 'image_2')
    train_gt_data_dir = os.path.join(args.data_dir, 'training', 'gt_image_2')
    test_data_dir = os.path.join(args.data_dir, 'testing', 'image_2')

    train_val_files = os.listdir(train_data_dir)
    train_val_gt_files= os.listdir(train_gt_data_dir)
    test_files = os.listdir(test_data_dir)

    random.seed(SEED)
    train_val_files.sort()
    random.shuffle(train_val_files)

    split_index = int(SPLIT * len(train_val_files))
    train_files, val_files = train_val_files[:split_index], train_val_files[split_index:]

    train_gt_files = [utils.data_file_to_gt(train_file) for train_file in train_files]
    val_gt_files = [utils.data_file_to_gt(val_file) for val_file in val_files]

    os.mkdir(args.output_dir)

    train_output_dir = os.path.join(args.output_dir, 'train')
    val_output_dir = os.path.join(args.output_dir, 'val')
    test_output_dir = os.path.join(args.output_dir, 'test')

    for output_dir in [train_output_dir, val_output_dir, test_output_dir]:
        os.mkdir(output_dir)

    data_map = [(train_data_dir, train_files, train_output_dir),
                (train_gt_data_dir, train_gt_files, train_output_dir),
                (train_data_dir, val_files, val_output_dir),
                (train_gt_data_dir, val_gt_files, val_output_dir),
                (test_data_dir, test_files, test_output_dir)]

    for data_dir, data_files, output_dir in data_map:
        for data_file in tqdm(data_files):
            from_path = os.path.join(data_dir, data_file)
            shutil.copy(from_path, output_dir)


