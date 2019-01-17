from __future__ import print_function, division

from keras.layers import Input
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import argparse
import pickle
import sys
try:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
except ImportError:
    print('Failed to import matplotlib!')
    pass
import os
import numpy as np


class Autoencoder():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoded_shape = (4, 4, 128)
        self.history = {'ae_loss': [], 'ae_acc': [], 'ae_test_loss': [], 'ae_test_acc': []}

        optimizer = Adam(0.001, 0.5)

        # Build and compile the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        self.autoencoder = Model(img, reconstructed_img)
        self.autoencoder.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        self.autoencoder.summary()
        print(self.autoencoder.metrics_names)

    def build_encoder(self):
        # Encoder
        encoder = Sequential()
        encoder.add(Conv2D(16, kernel_size=6, strides=1, padding='same', input_shape=self.img_shape))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(16, kernel_size=5, strides=2, padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(32, kernel_size=4, strides=2, padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(128, kernel_size=2, strides=2, padding='same'))

        encoder.summary()

        return encoder

    def build_decoder(self):
        # Decoder
        decoder = Sequential()
        decoder.add(Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', input_shape=self.encoded_shape))
        decoder.add(BatchNormalization())
        decoder.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Conv2DTranspose(16, kernel_size=4, strides=2, padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Conv2DTranspose(16, kernel_size=5, strides=2, padding='same'))
        decoder.add(BatchNormalization())
        decoder.add(Conv2DTranspose(3, kernel_size=6, strides=1, padding='same'))
        decoder.add(Activation(activation='tanh'))

        decoder.summary()

        return decoder

    def train(self, epochs, batch_size=128, sample_epoch=1, sample_interval=50, train_prop=0.8):

        # Load the dataset
        dataset = np.load('all.npy')
        mean = np.mean(dataset, axis=(0, 1, 2, 3))
        std = np.std(dataset, axis=(0, 1, 2, 3))
        dataset = (dataset.astype(np.float32) - mean) / (std + 1e-7)
        X_train = dataset[np.arange(0, int(np.floor(dataset.shape[0]*train_prop)))]
        X_test = dataset[np.arange(int(np.floor(dataset.shape[0]*train_prop)), dataset.shape[0])]
        iterations = int(np.ceil(X_train.shape[0] / batch_size))
        print('Start training on {} images and {} test images'.format(X_train.shape[0], X_test.shape[0]))
        print('There is a total of {} iterations per epoch'.format(iterations))
        if os.path.isfile('models/ae_autoencoder.h5'):
            self.autoencoder.load_weights('models/ae_autoencoder.h5')
            print('Loaded autoencoder weights!')

        for ep in range(epochs):
            # ---------------------
            #  Train Autoencoder
            # ---------------------
            index = np.arange(X_train.shape[0])
            for it in range(iterations):
                imgs_index = np.random.choice(index, min(batch_size, len(index)))
                index = np.delete(index, imgs_index)
                imgs = X_train[imgs_index]
                test_imgs = X_test[np.random.randint(0, X_test.shape[0], batch_size)]
                ae_loss = self.autoencoder.train_on_batch(imgs, imgs)
                ae_test_loss = self.autoencoder.test_on_batch(test_imgs, test_imgs)
                self.history['ae_loss'].append(ae_loss[0])
                self.history['ae_acc'].append(ae_loss[1])
                self.history['ae_test_loss'].append(ae_test_loss[0])
                self.history['ae_test_acc'].append(ae_test_loss[1])

                print('[Training Autoencoder AE]--- Epoch: {}/{} | It {}/{} | loss: {:.4f} | acc: {:.2f} | '
                      'test_loss: {:.4f} | test_acc: {:.2f}'.format(ep + 1, epochs, it, iterations, ae_loss[0],
                                                                    ae_loss[-1]*100, ae_test_loss[0],
                                                                    ae_test_loss[-1]*100), end='\r', flush=True)

            # If at save interval => save generated image samples
            if ep % sample_epoch == 0:
                # Select some images to see how the reconstruction gets better
                idx = np.arange(0, 25)
                imgs = X_train[idx]
                self.sample_images(ep, imgs)
                test_imgs = X_test[idx]
                self.sample_images(ep, test_imgs, plot='test')

    def plot(self):
        plt.figure()
        plt.title('Loss History')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        step = len(self.history['ae_loss']) // 10 if len(self.history['ae_loss']) > 100000 else 1
        plt.plot(np.arange(len(self.history['ae_loss'][::step])), self.history['ae_loss'][::step],
                 c='C0', label='train')
        plt.plot(np.arange(len(self.history['ae_test_loss'][::step])), self.history['ae_test_loss'][::step],
                 c='C1', label='test')
        plt.legend()
        plt.grid()
        plt.savefig('figs/ae_loss')

        plt.figure()
        plt.title('Acc History')
        plt.xlabel('Iter')
        plt.ylabel('Acc')
        step = len(self.history['ae_acc']) // 10 if len(self.history['ae_acc']) > 100000 else 1
        plt.plot(np.arange(len(self.history['ae_acc'][::step])), self.history['ae_acc'][::step], c='C0',
                 label='train')
        plt.plot(np.arange(len(self.history['ae_test_acc'][::step])), self.history['ae_test_acc'][::step], c='C1',
                 label='test')
        plt.legend()
        plt.grid()
        plt.savefig('figs/ae_accuracy')

    def sample_images(self, it, imgs, plot='train'):
        r, c = 5, 5

        if not os.path.isdir('images'):
            os.mkdir('images')

        gen_imgs = self.autoencoder.predict(imgs)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('images/{}_ae_{}.png'.format(plot, it))
        plt.close()

    def save_model(self):
        self.autoencoder.save_weights('models/ae_autoencoder.h5')
        with open('models/ae_history.pkl', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--epochs', type=int, help='number of epochs to train', default=100)
    parser.add_argument('--train_prop', type=float, help='Proportion of train set', default=0.8)
    return parser.parse_args(argv)


if __name__ == '__main__':
    ae = Autoencoder()
    args = parse_arguments(sys.argv[1:])
    print('Arguments: Epochs {}, batch_size {}, train_prop {}'.format(args.epochs, args.epochs, args.train_prop))
    try:
        ae.train(epochs=args.epochs, batch_size=args.batch_size, train_prop=args.train_prop)
        ae.save_model()
        ae.plot()
    except KeyboardInterrupt:
        ae.plot()
