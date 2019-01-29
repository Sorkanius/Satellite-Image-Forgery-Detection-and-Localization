from autoencoder import *
from PIL import Image
from classic_adversarial_autoencoder import *
from adversarial_autoencoder import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle


aae = AdversarialAutoencoder()
aae.autoencoder.load_weights('models/aae_autoencoder.h5')

svm = pickle.load(open('models/aae_svm.pkl', 'rb'))


images = os.listdir('data/forged_patches')

for img in images:
    image = np.array(Image.open('data/forged_patches/' + img), dtype='uint8')
    im = np.expand_dims(image/255, axis=0)
    rec = aae.autoencoder.predict(im)
    fig, axs = plt.subplots(1, 2)
    encoding = aae.encoder.predict(im)
    result = svm.predict(encoding.reshape(1, 2048))
    axs[0].imshow(image)
    axs[1].imshow(rec[0])
    axs[0].axis('off')
    axs[1].axis('off')
    fig.savefig('reconstructed_patches/' + ('Forged-' if result[0] == -1 else 'Pristine-') + img)
    plt.close()
