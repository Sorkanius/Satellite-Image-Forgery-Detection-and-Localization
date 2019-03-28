import numpy as np
import os
from PIL import Image
from skimage.measure import compare_ssim
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tqdm
import scipy.misc


def crop(im, height=64, width=64, stride=1):
    img_height = im.shape[0]
    img_width = im.shape[1]
    ssim = []
    for i in range(0, (img_height - height)//stride + 1):
        img = np.array([])
        for j in range(0, (img_width - width)//stride + 1):
            img = np.append(img, im[i*stride: i*stride + height, j*stride: j*stride + width])
        img = img.reshape((j+1), height, width, 3)
        re_im = aae.autoencoder.predict(img/255, batch_size=128)
        ssim.append([compare_ssim(img[l]/255, re_im[l], multichannel=True) for l in range(img.shape[0])])
    return ssim, j+1, i+1


def window(img_height, img_width, height=64, width=64, stride=1):
    wind = np.zeros([img_height, img_width])
    ones = np.ones([height, width])
    for i in range(0, (img_height - height)//stride + 1):
        for j in range(0, (img_width - width)//stride + 1):
            wind[i*stride: i*stride + height, j*stride: j*stride + width] = \
                wind[i*stride: i*stride + height, j*stride: j*stride + width] + ones
    return wind


def mse(image_A, image_B):
    err = np.sum((image_A.astype("float") - image_B.astype("float")) ** 2)
    err /= float(image_A.shape[0] * image_A.shape[1])
    return err


images_folder = 'data/train_data/'
images = os.listdir(images_folder)
threshold = 0.9777527527527528


from adversarial_autoencoder import *

aae = AdversarialAutoencoder()
aae.autoencoder.load_weights('trainings/no_gan/models/low_ae_autoencoder.h5')

total_predictions = []
wind = window(640, 640, stride=8)
plt.imshow(wind)
plt.savefig('window.png')
plt.close()

for image in tqdm.tqdm(images):
    im_read = np.array(Image.open(os.path.join(images_folder, image)), dtype='uint8')
    ssim, rows, cols = crop(im_read, stride=8)
    prediction = np.zeros([640, 640])
    ones = np.ones([64, 64])
    ssim = np.asarray(ssim)
    ssim = ssim.reshape(rows, cols)
    attack_prop = range(10, 110, 10)
    mse_of_attack_ones = []
    mse_of_attack_zeros = []
    mse_of_attack_mix = []
    for i in range(0, (640 - 64)//8 + 1):
        for j in range(0, (640 - 64)//8 + 1):
            if ssim[i, j] <= threshold:
                prediction[i*8: i*8 + 64, j*8: j*8 + 64] = prediction[i*8: i*8 + 64, j*8: j*8 + 64] + ones
        fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im_read)
    axs[1].imshow(prediction/wind)
    axs[0].axis('off')
    axs[1].axis('off')
    fig.savefig('postprocessing/pred_{}.png'.format(image))
    plt.close()
    print('saved')
    total_predictions.append(prediction)



