import numpy as np
from PIL import Image
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to img folder', required=True)
    parser.add_argument('--output', help='name of output file', default='data.npy')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    data = []
    output = args.output
    images_folder = args.input
    images = os.listdir(images_folder)

    [data.append(np.array(Image.open(images_folder + img), dtype='uint8')) for img in images]
    np.save(args.output, data)

