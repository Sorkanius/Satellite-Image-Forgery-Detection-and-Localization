from __future__ import print_function, division

from keras.layers import Input, Dense, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
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


class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoded_shape = (4, 4, 128)
        self.history = {'d_loss': [], 'd_acc': [], 'g_loss': [], 'g_acc': []}

        optimizer = Adam(0.0005, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        self.autoencoder = Model(img, reconstructed_img)
        self.autoencoder.compile(loss='mse', optimizer=optimizer)
        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(reconstructed_img)

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

        model = Sequential()

        model.add(Conv2D(16, kernel_size=5, strides=1, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        
        model.add(Conv2D(16, kernel_size=2, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        
        model.add(Conv2D(32, kernel_size=4, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        
        model.add(Conv2D(32, kernel_size=2, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=3, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64, kernel_size=2, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        model.add(Activation(activation='sigmoid'))
        model.summary()

        return model

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
        plt.grid()
        plt.savefig('figs/pretrain_ae')
        
        self.autoencoder.save_weights('models/aae_autoencoder.h5')

    def pretrain_discriminator(self, data, iterations, batch_size):
        half_batch = batch_size // 2
        fake = np.zeros((half_batch, 1))
        valid = np.ones((half_batch, 1))
        history = {'loss': [], 'acc': []}

        for it in range(iterations):
            idx = np.random.randint(0, data.shape[0], half_batch)
            imgs = data[idx]
            generated_imgs = self.autoencoder.predict(imgs)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(generated_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            history['loss'].append(d_loss[0])
            history['acc'].append(d_loss[1])
            print('[Pretrain Discriminator]---it {}/{} | loss: {:.4f} | acc {:.2f}'
                  .format(it, iterations, d_loss[0], d_loss[1]), end='\r', flush=True)
        
        plt.figure()
        plt.title('Pretrain Discriminator')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        step = len(history['loss']) // 10 if len(history['loss']) > 1000 else 1
        plt.plot(np.arange(len(history['loss'][::step])), history['loss'][::step])
        plt.grid()
        plt.savefig('figs/pretrain_discriminator')

        self.discriminator.save_weights('models/aae_discriminator.h5')

    def train(self, epochs, pre_dis_iterations, pre_ae_iterations, batch_size=128,
              sample_epoch=1, sample_interval=50):

        # Load the dataset
        X_train = np.load('data.npy')
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train.astype(np.float32) - mean) / (std + 1e-7)
        iterations = int(np.ceil(X_train.shape[0] / batch_size))
        print('Start training on {} images'.format(X_train.shape[0]))
        print('There is a total of {} iterations per epoch'.format(iterations))

        if os.path.isfile('models/aae_discriminator.h5'):
            self.discriminator.load_weights('models/aae_discriminator.h5')
            print('Loaded discriminator weights!')
        elif pre_dis_iterations > 0:
            self.pretrain_discriminator(X_train, pre_dis_iterations, batch_size)

        if os.path.isfile('models/aae_autoencoder.h5'):
            self.autoencoder.load_weights('models/aae_autoencoder.h5')
            print('Loaded autoencoder weights!')
        elif pre_ae_iterations > 0:
            self.pretrain_ae(X_train, pre_ae_iterations, batch_size)

        for ep in range(epochs):
            index_dis = np.arange(X_train.shape[0])
            index_gen = np.arange(X_train.shape[0])
            for it in range(iterations):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                imgs_index_dis = np.random.choice(index_dis, min(batch_size, len(index_dis)))
                index_dis = np.delete(index_dis, imgs_index_dis)
                imgs = X_train[imgs_index_dis]
                generated_imgs = self.autoencoder.predict(imgs)

                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((min(batch_size, len(index_dis)), 1)))
                d_loss_fake = self.discriminator.train_on_batch(generated_imgs,
                                                                np.zeros((min(batch_size, len(index_dis)), 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                self.history['d_loss'].append(d_loss[0])
                self.history['d_acc'].append(d_loss[1] * 100)

                # ---------------------
                #  Train Generator
                # ---------------------
                imgs_index_gen = np.random.choice(index_gen, min(batch_size, len(index_gen)))
                index_gen = np.delete(index_gen, imgs_index_gen)
                imgs = X_train[imgs_index_gen]
                g_loss = self.adversarial_autoencoder.train_on_batch(imgs,
                                                                     [imgs,
                                                                      np.ones((min(batch_size, len(index_gen)), 1))])
                self.history['g_loss'].append(g_loss[0])
                self.history['g_acc'].append(g_loss[-1]*100)

                print('[Training Adversarial AE]--- Epoch: {}/{} | It {}/{} |'
                      ' d_loss: {:.4f} | d_acc: {:.2f} |'
                      ' g_loss: {:.4f} | g_acc: {:.2f}'
                      .format(ep + 1, epochs, it, iterations, d_loss[0], d_loss[1]*100, g_loss[0], g_loss[-1]*100), end='\r', flush=True)

            # If at save interval => save generated image samples
            if ep % sample_epoch == 0:
                # Select a random half batch of images
                idx = np.arange(0, 25)
                imgs = X_train[idx]
                self.sample_images(ep, imgs)

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
        plt.grid()
        plt.savefig('figs/aae_loss')

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
        plt.grid()
        plt.savefig('figs/aae_accuracy')

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
        fig.savefig('images/aae_%d.png' % it)
        plt.close()

    def save_model(self):
        self.adversarial_autoencoder.save_weights('models/aae_adversarial_ae.h5')
        self.discriminator.save_weights('models/aae_discriminator.h5')
        self.autoencoder.save_weights('models/aae_autoencoder.h5')
        with open('models/aae_history.pkl', 'wb') as f:
                pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--epochs', type=int, help='number of iterations to train', default=100)
    parser.add_argument('--ae_it', type=int,
                        help='number of iterations to pretrain the autoencoder', default=0)
    parser.add_argument('--d_it', type=int,
                        help='number of iterations to pretrain the discriminator', default=0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    args = parse_arguments(sys.argv[1:])
    print('Arguments: epochs {}, pre_ae_iterations {}, pre_dis_iterations {}, batch_size {}'.format(
        args.epochs, args.ae_it, args.d_it, args.batch_size))
    try:
        aae.train(epochs=args.epochs, pre_ae_iterations=args.ae_it,
                  pre_dis_iterations=args.d_it, batch_size=args.batch_size)
        aae.save_model()
        aae.plot()
    except KeyboardInterrupt:
        aae.plot()
