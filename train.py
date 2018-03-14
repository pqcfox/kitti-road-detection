import argparse
import os

import tensorflow as tf

SEED = 230

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':
    tf.set_random_seed(230)

    # Load parameters

    # Logging utilities?

    # Create pipeline
