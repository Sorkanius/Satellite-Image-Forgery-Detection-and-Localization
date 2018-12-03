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
        self.history = {'rec_loss': [], 'rec_acc': [], 'rec_test_loss': [], 'rec_test_acc': [],
                        'reg_loss': [], 'reg_acc': [], 'reg_test_loss': [], 'reg_test_acc': []}

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
        discriminator.add(Dense(128, activation="relu", input_shape=(self.latent_shape,)))
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
        step = len(history['loss']) // 10 if len(history['loss']) > 100000 else 1
        plt.plot(np.arange(len(history['loss'][::step])), history['loss'][::step])
        plt.grid()
        plt.savefig('figs/c_aae_pretrain_ae')

        self.autoencoder.save_weights('models/c_aae_autoencoder.h5')

    def train(self, epochs, pre_ae_iterations, batch_size=128, sample_epoch=1, sample_interval=50, train_prop=0.8):

        # Load the dataset
        dataset = np.load('data.npy')
        mean = np.mean(dataset, axis=(0, 1, 2, 3))
        std = np.std(dataset, axis=(0, 1, 2, 3))
        dataset = (dataset.astype(np.float32) - mean) / (std + 1e-7)
        X_train = dataset[np.arange(0, int(np.floor(dataset.shape[0]*train_prop)))]
        X_test = dataset[np.arange(int(np.floor(dataset.shape[0]*train_prop)), dataset.shape[0])]
        iterations = int(np.ceil(X_train.shape[0] / batch_size))
        print('Start training on {} images and {} test images'.format(X_train.shape[0], X_test.shape[0]))
        print('There is a total of {} iterations per epoch'.format(iterations))

        if os.path.isfile('models/c_aae_autoencoder.h5'):
            self.autoencoder.load_weights('models/c_aae_autoencoder.h5')
            print('Loaded autoencoder weights!')
        elif pre_ae_iterations > 0:
            self.pretrain_ae(X_train, pre_ae_iterations, batch_size)

        for ep in range(epochs):
            index_reg = np.arange(X_train.shape[0])
            index_rec = np.arange(X_train.shape[0])
            for it in range(iterations):
                # ---------------------
                #  Regularization Phase
                # ---------------------
                imgs_index_reg = np.random.choice(index_reg, min(batch_size, len(index_reg)))
                index_reg = np.delete(index_reg, min(batch_size, len(index_reg)))
                imgs = X_train[imgs_index_reg]
                test_imgs = X_test[np.random.randint(0, X_test.shape[0], batch_size)]
                latent_fake = self.encoder.predict(imgs)
                latent_fake_test = self.encoder.predict(test_imgs)
                latent_real = np.asarray(self.generate_p_sample(batch_size))
                latent_real_test = np.asarray(self.generate_p_sample(batch_size))

                reg_loss_real = self.discriminator.train_on_batch(latent_real,
                                                                  np.ones((min(batch_size, len(index_reg)), 1)))
                reg_loss_fake = self.discriminator.train_on_batch(latent_fake.reshape(batch_size, self.latent_shape),
                                                                  np.zeros((min(batch_size, len(index_reg)), 1)))
                reg_loss = 0.5 * np.add(reg_loss_real, reg_loss_fake)

                reg_test_loss_real = self.discriminator.test_on_batch(latent_real_test, np.ones((batch_size, 1)))
                reg_test_loss_fake = self.discriminator.test_on_batch(latent_fake_test.reshape(batch_size,
                                                                                               self.latent_shape),
                                                                      np.zeros((batch_size, 1)))
                reg_test_loss = 0.5 * np.add(reg_test_loss_real, reg_test_loss_fake)
                self.history['reg_loss'].append(reg_loss[0])
                self.history['reg_acc'].append(reg_loss[1] * 100)
                self.history['reg_test_loss'].append(reg_test_loss[0])
                self.history['reg_test_acc'].append(reg_test_loss[1] * 100)

                # ---------------------
                #  Reconstruction Phase
                # ---------------------
                imgs_index_rec = np.random.choice(index_rec, min(batch_size, len(index_rec)))
                index_rec = np.delete(index_rec, imgs_index_rec)
                imgs = X_train[imgs_index_rec]
                test_imgs = X_test[np.random.randint(0, X_test.shape[0], batch_size)]
                rec_loss = self.adversarial_autoencoder.train_on_batch(imgs,
                                                                       [imgs, np.ones((min(batch_size, len(index_rec)),
                                                                                       1))])
                rec_test_loss = self.adversarial_autoencoder.test_on_batch(test_imgs,
                                                                           [test_imgs, np.ones((batch_size, 1))])
                self.history['rec_loss'].append(rec_loss[0])
                self.history['rec_acc'].append(rec_loss[-1]*100)
                self.history['rec_test_loss'].append(rec_test_loss[0])
                self.history['rec_test_acc'].append(rec_test_loss[-1]*100)

                print('[Training Adversarial AE]--- Epoch: {}/{} | It {}/{} | '
                      'reg_loss: {:.4f} | reg_acc: {:.2f} | '
                      'rec_loss: {:.4f} | rec_acc: {:.2f} | '
                      'reg_test_loss: {:.4f} | reg_test_acc: {:.2f} | '
                      'rec_test_loss: {:.4f} | rec_test_acc: {:.2f} | '
                      .format(ep + 1, epochs, it, iterations, reg_loss[0],
                              reg_loss[1] * 100, rec_loss[0], rec_loss[-1] * 100, reg_test_loss[0],
                              reg_test_loss[1] * 100, rec_test_loss[0], rec_test_loss[-1] * 100), end='\r', flush=True)

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
        step = len(self.history['rec_loss']) // 10 if len(self.history['rec_loss']) > 100000 else 1
        plt.plot(np.arange(len(self.history['rec_loss'][::step])), self.history['rec_loss'][::step],
                 c='C0', label='reconstruction loss')
        plt.plot(np.arange(len(self.history['reg_loss'][::step])), self.history['reg_loss'][::step],
                 c='C1', label='regularization loss')
        plt.plot(np.arange(len(self.history['rec_test_loss'][::step])), self.history['rec_test_loss'][::step],
                 c='C2', label='reconstruction test loss')
        plt.plot(np.arange(len(self.history['reg_test_loss'][::step])), self.history['reg_test_loss'][::step],
                 c='C3', label='regularization test loss')
        plt.legend()
        plt.grid()
        plt.savefig('figs/c_aae_loss')

        plt.figure()
        plt.title('Acc History')
        plt.xlabel('Iter')
        plt.ylabel('Acc')
        step = len(self.history['rec_acc']) // 10 if len(self.history['rec_acc']) > 100000 else 1
        plt.plot(np.arange(len(self.history['rec_acc'][::step])), self.history['rec_acc'][::step], c='C0',
                 label='reconstruction')
        plt.plot(np.arange(len(self.history['reg_acc'][::step])), self.history['reg_acc'][::step], c='C1',
                 label='regularization')
        plt.plot(np.arange(len(self.history['rec_test_acc'][::step])), self.history['rec_test_acc'][::step], c='C2',
                 label='reconstruction test')
        plt.plot(np.arange(len(self.history['reg_test_acc'][::step])), self.history['reg_test_acc'][::step], c='C3',
                 label='regularization test')
        plt.legend()
        plt.grid()
        plt.savefig('figs/c_aae_accuracy')

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
        fig.savefig('images/{}_c_aae_{}.png'.format(plot, it))
        plt.close()

    def save_model(self):
        self.adversarial_autoencoder.save_weights('models/c_aae_adversarial_ae.h5')
        self.discriminator.save_weights('models/c_aae_discriminator.h5')
        self.autoencoder.save_weights('models/c_aae_autoencoder.h5')
        with open('models/c_aae_history.pkl', 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--epochs', type=int, help='number of iterations to train', default=100)
    parser.add_argument('--ae_it', type=int,
                        help='number of epochs to pretrain the autoencoder', default=0)
    parser.add_argument('--train_prop', type=int, help='Proportion of train set', default=0.8)
    return parser.parse_args(argv)


if __name__ == '__main__':
    c_aae = ClassicAdversarialAutoencoder()
    args = parse_arguments(sys.argv[1:])
    print('Arguments: iterations {}, pre_ae_iterations {}, batch_size {}, train_prop {}'.format(
        args.epochs, args.ae_it, args.batch_size, args.train_prop))
    try:
        c_aae.train(epochs=args.epochs, pre_ae_iterations=args.ae_it, batch_size=args.batch_size,
                    train_prop=args.train_prop)
        c_aae.save_model()
        c_aae.plot()
    except KeyboardInterrupt:
        c_aae.plot()
