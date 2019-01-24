import tensorflow as tf
import os
import numpy as np
import argparse
import sys
from tqdm import tqdm
import cv2
from autoencoder import *
from PIL import Image
from classic_adversarial_autoencoder import *
from adversarial_autoencoder import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


ae = Autoencoder()
ae.autoencoder.load_weights('models/ae_autoencoder.h5')

images = os.listdir('data/forged_patches')
img = images[0]
for img in images:
    im = np.array(Image.open('data/forged_patches/' + img), dtype='uint8')
    rec = ae.autoencoder.predict(np.expand_dims(im/255, axis=0))
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(im)
    # axs[1].imshow((rec[0] * 255).astype(np.uint8))
    axs[1].imshow(rec[0])
    axs[0].axis('off')
    axs[1].axis('off')
    fig.savefig('reconstructed_patches/' + img)
    plt.close()
