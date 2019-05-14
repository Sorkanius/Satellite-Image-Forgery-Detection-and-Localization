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

wind = window(640, 640, stride=8)
plt.imshow(wind)
plt.savefig('window.png')
plt.close()

attack_prop = range(0, 110, 10)

mse_of_attack_ones = {}
mse_of_attack_zeros = {}
mse_of_attack_mix = {}
mse_of_one_node = []

all_zeros = np.zeros([640, 640])

for i in attack_prop:
    mse_of_attack_ones[i] = []
    mse_of_attack_zeros[i] = []
    mse_of_attack_mix[i] = []

for image in tqdm.tqdm(images):
    im_read = np.array(Image.open(os.path.join(images_folder, image)), dtype='uint8')
    ssim, rows, cols = crop(im_read, stride=8)
    prediction = np.zeros([640, 640])
    ones = np.ones([64, 64])
    ssim = np.asarray(ssim)
    ssim = ssim.reshape(rows, cols)
    new_dir = 'postprocessing_with_attacks/{}/'.format(image)
    os.makedirs(new_dir, exist_ok=True)
    for i in range(0, (640 - 64)//8 + 1):
        for j in range(0, (640 - 64)//8 + 1):
            if ssim[i, j] <= threshold:
                prediction[i*8: i*8 + 64, j*8: j*8 + 64] = prediction[i*8: i*8 + 64, j*8: j*8 + 64] + ones
    mse_of_one_node.append(mse(all_zeros, prediction/wind))
    for k in attack_prop:
        attacks = np.random.binomial(1, k/100, [rows, cols])
        prediction_ones = np.zeros([640, 640])
        prediction_zeros = np.zeros([640, 640])
        prediction_mix = np.zeros([640, 640])
        for i in range(0, (640 - 64)//8 + 1):
            for j in range(0, (640 - 64)//8 + 1):
                if attacks[i, j] == 0:
                    if ssim[i, j] <= threshold:
                        prediction_ones[i*8: i*8 + 64, j*8: j*8 + 64] = \
                            prediction_ones[i*8: i*8 + 64, j*8: j*8 + 64] + ones
                        prediction_zeros[i*8: i*8 + 64, j*8: j*8 + 64] = \
                            prediction_zeros[i*8: i*8 + 64, j*8: j*8 + 64] + ones
                        prediction_mix[i*8: i*8 + 64, j*8: j*8 + 64] = \
                            prediction_mix[i*8: i*8 + 64, j*8: j*8 + 64] + ones
                else:
                    prediction_ones[i*8: i*8 + 64, j*8: j*8 + 64] = \
                        prediction_ones[i*8: i*8 + 64, j*8: j*8 + 64] + ones
                    prediction_zeros[i*8: i*8 + 64, j*8: j*8 + 64] = \
                        prediction_zeros[i*8: i*8 + 64, j*8: j*8 + 64]
                    prediction_mix[i*8: i*8 + 64, j*8: j*8 + 64] = \
                        prediction_mix[i*8: i*8 + 64, j*8: j*8 + 64] + np.random.binomial(1, 0.5, 1)[0]*ones
        mse_of_attack_ones[k].append(mse(all_zeros, prediction_ones/wind))
        mse_of_attack_zeros[k].append(mse(all_zeros, prediction_zeros/wind))
        mse_of_attack_mix[k].append(mse(all_zeros, prediction_mix/wind))
        fig, axs = plt.subplots(1, 5)
        axs[0].imshow(im_read)
        axs[1].imshow(prediction/wind)
        axs[2].imshow(prediction_ones/wind, vmin=0.0, vmax=1.0)
        axs[3].imshow(prediction_zeros/wind)
        axs[4].imshow(prediction_mix/wind)
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')
        axs[4].axis('off')
        fig.savefig(new_dir + 'attack_{}.png'.format(k))
        plt.close()

mean_ones = []
mean_zeros = []
mean_mix = []
for i in sorted(mse_of_attack_ones.keys()):
    mean_ones.append(np.mean(mse_of_attack_ones[i]))
    mean_zeros.append(np.mean(mse_of_attack_zeros[i]))
    mean_mix.append(np.mean(mse_of_attack_mix[i]))


plt.figure()
plt.xlabel('% of attackers')
plt.ylabel('MSE')
plt.semilogy(list(attack_prop), mean_ones, c='C0', label='Always One attack')
plt.semilogy(list(attack_prop), mean_zeros, c='C1', label='Always Zero attack')
plt.semilogy(list(attack_prop), mean_mix, c='C2', label='Random attack')
plt.semilogy(list(attack_prop), np.repeat(np.mean(mse_of_one_node), len(list(attack_prop))), c='C4',
             label='Centralized Node')
plt.legend()
plt.grid()
plt.savefig('postprocessing_with_attacks/mse.png')
plt.close()


