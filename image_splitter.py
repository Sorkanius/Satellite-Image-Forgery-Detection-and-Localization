import matplotlib
import numpy as np
import os
import argparse
from PIL import Image
matplotlib.use('agg')
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to img folder', required=True)
    parser.add_argument('--output', help='path to the output patch folder', default='data/patches')
    parser.add_argument('--size', help='Patch size', type=int, default=64)

    return parser.parse_args()


def crop(im, height, width, path, filename):
    img_height = im.shape[0]
    img_width = im.shape[1]

    if (img_height != 640) or (img_width != 640):
        return False

    for i in range(0, img_height//height - 1):
        for j in range(0, img_width//width): # last row is removed for avoiding watermarks
            crop = im[i*height:(i+1)*height, j*width:(j+1)*width]
            Image.fromarray(crop.astype('uint8'), 'RGB').save(os.path.join(path, '{}_{}_{}'.format(i, j, filename)))


if __name__ == '__main__':
    args = parse_arguments()
    data = []
    output = args.output
    images_folder = args.input
    images = os.listdir(images_folder)
    for im in images:
        img = np.array(Image.open(os.path.join(images_folder, im)), dtype='uint8')
        crop(img, 64, 64, args.output, im)
