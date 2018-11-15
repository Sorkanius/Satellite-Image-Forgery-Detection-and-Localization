from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten
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


class ClassicAdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoded_shape = (4, 4, 128)
        self.latent_shape = self.encoded_shape[0] * self.encoded_shape[1] * self.encoded_shape[2]
        self.history = {'rec_loss': [], 'rec_acc': [], 'reg_loss': [], 'reg_acc': []}

        optimizer = Adam(0.0005, 0.5)

        # Build discriminator
        self.discriminator = self.build_discriminator()

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)

        # The autoencoder takes the image, encodes it and reconstructs it from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        self.autoencoder = Model(img, reconstructed_img)
        self.autoencoder.compile(loss='mse', optimizer=optimizer)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(Flatten()(encoded_repr))

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
                                             loss_weights=[0.99, 0.01], optimizer=optimizer, metrics=['accuracy'])
        self.adversarial_autoencoder.summary()
        print(self.adversarial_autoencoder.metrics_names)

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

    def build_discriminator(self):
        # Discriminador
        discriminator = Sequential()
        discriminator.add(Dense(128, activation="relu", input_shape=(1,)))
        discriminator.add(Dense(128, activation="relu"))
        discriminator.add(Dense(1, activation="sigmoid"))

        discriminator.summary()

        return discriminator

    def generate_p_sample(self, batch_size, mean=0, stddev=1):
        return [np.random.normal(mean, stddev, self.latent_shape) for _ in range(batch_size)]

    def pretrain_ae(self, data, iterations, batch_size):
        history = {'loss': []}

        for it in range(iterations):
            idx = np.random.randint(0, data.shape[0], batch_size)
            imgs = data[idx]
            train_loss = self.autoencoder.train_on_batch(imgs, imgs)
            history['loss'].append(train_loss)

            print('[Pretrain AE]---It {}/{} | AE loss: {:.4f}'.format(it, iterations, train_loss), end='\r', flush=True)

        plt.figure()
        plt.title('Pretrain AE')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        step = len(history['loss']) // 10 if len(history['loss']) > 1000 else 1
        plt.plot(np.arange(len(history['loss'][::step])), history['loss'][::step])
        plt.savefig('c_aae pretrain ae')

        self.autoencoder.save_weights('c_aae_autoencoder.h5')

    def train(self, iterations, pre_ae_iterations, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = np.load('data.npy')
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train.astype(np.float32) - mean) / (std + 1e-7)
        print('Start training on {} images'.format(X_train.shape[0]))

        if os.path.isfile('c_aae_autoencoder.h5'):
            self.autoencoder.load_weights('c_aae_autoencoder.h5')
            print('Loaded autoencoder weights!')
        elif pre_ae_iterations > 0:
            self.pretrain_ae(X_train, pre_ae_iterations, batch_size)

        valid = np.ones((batch_size, 1))
        half_batch = batch_size // 2
        half_valid = np.ones((half_batch, 1))
        half_fake = np.zeros((half_batch, 1))

        for it in range(iterations):
            # ---------------------
            #  Regularization Phase
            # ---------------------
            imgs = X_train[np.random.randint(0, X_train.shape[0], half_batch)]
            latent_fake = self.encoder.predict(imgs)
            latent_real = self.generate_p_sample(half_batch)

            reg_loss_real = self.discriminator.train_on_batch(latent_real, half_valid)
            reg_loss_fake = self.discriminator.train_on_batch(latent_fake, half_fake)
            reg_loss = 0.5 * np.add(reg_loss_real, reg_loss_fake)

            self.history['reg_loss'].append(reg_loss[0])
            self.history['reg_acc'].append(reg_loss[1] * 100)

            # ---------------------
            #  Reconstruction Phase
            # ---------------------
            imgs = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            rec_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])
            self.history['rec_loss'].append(rec_loss[0])
            self.history['rec_acc'].append(rec_loss[-1]*100)

            print('[Training Adversarial AE]---It {}/{} |'
                  ' reg_loss: {:.4f} | reg_acc: {:.2f} |'
                  ' rec_loss: {:.4f} | rec_acc: {:.2f}'
                  .format(it, iterations, reg_loss[0],
                          reg_loss[1] * 100, rec_loss[0], rec_loss[-1] * 100), end='\r', flush=True)

            # If at save interval => save generated image samples
            if it % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 25)
                imgs = X_train[idx]
                self.sample_images(it, imgs)

    def plot(self):
        plt.figure()
        plt.title('Loss History')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        step = len(self.history['d_loss']) // 10 if len(self.history['d_loss']) > 1000 else 1
        plt.plot(np.arange(len(self.history['d_loss'][::step])), self.history['d_loss'][::step],
                 c='C0', label='discriminator')
        plt.plot(np.arange(len(self.history['g_loss'][::step])), self.history['g_loss'][::step],
                 c='C1', label='generator')
        plt.legend()
        plt.savefig('c_aae_loss')

        plt.figure()
        plt.title('Acc History')
        plt.xlabel('Iter')
        plt.ylabel('Acc')
        step = len(self.history['d_acc']) // 10 if len(self.history['d_acc']) > 1000 else 1
        plt.plot(np.arange(len(self.history['d_acc'][::step])), self.history['d_acc'][::step], c='C0',
                 label='discriminator')
        plt.plot(np.arange(len(self.history['g_acc'][::step])), self.history['g_acc'][::step], c='C1',
                 label='generator')
        plt.legend()
        plt.savefig('c_aae_accuracy')

    def sample_images(self, it, imgs):
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
        fig.savefig('images/c_aae_%d.png' % it)
        plt.close()

    def save_model(self):
        self.adversarial_autoencoder.save_weights('c_aae_adversarial_ae.h5')
        self.discriminator.save_weights('c_aae_discriminator.h5')
        self.autoencoder.save_weights('c_aae_autoencoder.h5')
        with open('c_aae_history.pkl', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--it', type=int, help='number of iterations to train', default=10000)
    parser.add_argument('--ae_it', type=int,
                        help='number of epochs to pretrain the autoencoder', default=20000)
    return parser.parse_args(argv)


if __name__ == '__main__':
    c_aae = ClassicAdversarialAutoencoder()
    args = parse_arguments(sys.argv[1:])
    print('Arguments: iterations {}, pre_ae_iterations {}, batch_size {}'.format(
        args.it, args.ae_it, args.batch_size))
    try:
        c_aae.train(iterations=args.it, pre_ae_iterations=args.ae_it, batch_size=args.batch_size)
        c_aae.save_model()
        c_aae.plot()
    except KeyboardInterrupt:
        c_aae.plot()
