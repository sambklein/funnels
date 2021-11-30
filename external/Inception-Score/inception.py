# To use https://github.com/tsc2017/Inception-Score/blob/master/inception_score.py
import argparse
import glob
import os
import numpy as np
from imageio import imread

from inception_score import get_inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar-10', help='String name of the dataset')
    return parser.parse_args()


def get_images(dataset):
    if not os.path.isdir(dataset):
        raise Exception('path does not exist')
    files = list(glob.glob(f'{dataset}/*.jpg')) + list(glob.glob(f'{dataset}/*.png'))
    save_dataset = np.array([imread(str(fn)).astype(np.float32) for fn in files])
    return save_dataset * 255


if __name__ == '__main__':
    args = parse_args()
    print(f'Evaluating on {args.dataset}.')
    data = get_images(args.dataset)
    data = data.transpose(0, 3, 1, 2)
    print("Calculating Inception Score...")
    print(get_inception_score(data, splits=10))
