from __future__ import print_function, division

from sklearn.svm import OneClassSVM
from keras.layers import Input, Dense, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import json
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
import numpy as np


class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.encoded_shape = (4, 4, 128)
        self.history = {'d_loss': [], 'd_acc': [], 'd_test_loss': [], 'd_test_acc': [],
                        'g_loss': [], 'g_acc': [], 'g_test_loss': [], 'g_test_acc': []}

        optimizer = Adam(0.001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=SGD(0.001), metrics=['accuracy'])

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

        return model

    def train(self, epochs, X_train, X_test, batch_size=128):

        iterations = int(np.ceil(X_train.shape[0] / batch_size))
        print('Start training on {} images and {} test images'.format(X_train.shape[0], X_test.shape[0]))
        print('There is a total of {} iterations per epoch'.format(iterations))

        for ep in range(epochs):
            index_dis = np.arange(X_train.shape[0])
            index_gen = np.arange(X_train.shape[0])
            for it in range(iterations):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                imgs_index_dis = np.random.choice(index_dis, min(batch_size, len(index_dis)), replace=False)
                index_dis = np.delete(index_dis, imgs_index_dis)
                imgs = X_train[imgs_index_dis]
                test_imgs = X_test[np.random.randint(0, X_test.shape[0], batch_size)]

                generated_imgs = self.autoencoder.predict(imgs)
                generated_test_imgs = self.autoencoder.predict(test_imgs)

                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((min(batch_size, imgs.shape[0]), 1)))
                d_loss_fake = self.discriminator.train_on_batch(generated_imgs,
                                                                np.zeros((min(batch_size, imgs.shape[0]), 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                d_test_loss_real = self.discriminator.test_on_batch(test_imgs, np.ones((batch_size, 1)))
                d_test_loss_fake = self.discriminator.test_on_batch(generated_test_imgs, np.zeros((batch_size, 1)))
                d_test_loss = 0.5 * np.add(d_test_loss_real, d_test_loss_fake)
                self.history['d_loss'].append(d_loss[0])
                self.history['d_acc'].append(d_loss[1] * 100)
                self.history['d_test_loss'].append(d_test_loss[0])
                self.history['d_test_acc'].append(d_test_loss[1] * 100)

                # ---------------------
                #  Train Generator
                # ---------------------
                imgs_index_gen = np.random.choice(index_gen, min(batch_size, len(index_gen)), replace=False)
                index_gen = np.delete(index_gen, imgs_index_gen)
                imgs = X_train[imgs_index_gen]
                test_imgs = X_test[np.random.randint(0, X_test.shape[0], batch_size)]

                g_loss = self.adversarial_autoencoder.train_on_batch(imgs,
                                                                     [imgs,
                                                                      np.ones((min(batch_size, imgs.shape[0]), 1))])
                g_test_loss = self.adversarial_autoencoder.test_on_batch(test_imgs,
                                                                         [test_imgs, np.ones((batch_size, 1))])
                self.history['g_loss'].append(g_loss[0])
                self.history['g_acc'].append(g_loss[-1]*100)
                self.history['g_test_loss'].append(g_test_loss[0])
                self.history['g_test_acc'].append(g_test_loss[-1]*100)

                print('[Training Adversarial AE]--- Epoch: {}/{} | It {}/{} | '
                      'd_loss: {:.4f} | d_acc: {:.2f} | '
                      'g_loss: {:.4f} | g_acc: {:.2f} | '
                      'd_test_loss: {:.4f} | d_test_acc: {:.2f} | '
                      'g_test_loss: {:.4f} | g_test_acc: {:.2f} | '
                      .format(ep + 1, epochs, it, iterations, d_loss[0], d_loss[1]*100, g_loss[0], g_loss[-1]*100,
                              d_test_loss[0], d_test_loss[1]*100, g_test_loss[0], g_test_loss[-1]*100),
                      end='\r', flush=True)

        return g_loss[0], g_test_loss[0]

    def plot(self, id):
        plt.figure()
        plt.title('Loss History')
        plt.xlabel('Iter')
        plt.ylabel('Loss')
        step = len(self.history['d_loss']) // 10 if len(self.history['d_loss']) > 100000 else 1
        plt.plot(np.arange(len(self.history['d_loss'][::step])), self.history['d_loss'][::step],
                 c='C0', label='discriminator')
        plt.plot(np.arange(len(self.history['g_loss'][::step])), self.history['g_loss'][::step],
                 c='C1', label='generator')
        plt.plot(np.arange(len(self.history['d_test_loss'][::step])), self.history['d_test_loss'][::step],
                 c='C2', label='discriminator test')
        plt.plot(np.arange(len(self.history['g_test_loss'][::step])), self.history['g_test_loss'][::step],
                 c='C3', label='generator test')
        plt.legend()
        plt.grid()
        plt.savefig('nodes/figs/{}_loss'.format(id))

        plt.figure()
        plt.title('Acc History')
        plt.xlabel('Iter')
        plt.ylabel('Acc')
        step = len(self.history['d_acc']) // 10 if len(self.history['d_acc']) > 100000 else 1
        plt.plot(np.arange(len(self.history['d_acc'][::step])), self.history['d_acc'][::step], c='C0',
                 label='discriminator')
        plt.plot(np.arange(len(self.history['g_acc'][::step])), self.history['g_acc'][::step], c='C1',
                 label='generator')
        plt.plot(np.arange(len(self.history['d_test_acc'][::step])), self.history['d_test_acc'][::step], c='C2',
                 label='discriminator test')
        plt.plot(np.arange(len(self.history['g_test_acc'][::step])), self.history['g_test_acc'][::step], c='C3',
                 label='generator test')
        plt.legend()
        plt.grid()
        plt.savefig('nodes/figs/{}_accuracy'.format(id))

    def save_model(self, id):
        self.autoencoder.save_weights('nodes/autoencoders/{}.h5'.format(id))
        with open('nodes/history/{}.pkl'.format(id), 'wb') as f:
            pickle.dump(self.history, f, pickle.HIGHEST_PROTOCOL)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch_size', default=128)
    parser.add_argument('--epochs', type=int, help='number of iterations to train', default=50)
    parser.add_argument('--train_prop', type=float, help='Proportion of train set', default=0.8)
    parser.add_argument('--nodes', type=int, help='Number of nodes to train', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    n_nodes = args.nodes
    dataset = np.load('new_data.npy')/255
    forged = np.load('forged.npy')/255
    # epochs = abs(np.random.normal(0, 10, n_nodes) + args.epochs)
    # epochs = [int(x) for x in epochs]
    epochs = np.ones(n_nodes, dtype='int')
    x_train = dataset[np.arange(0, int(np.floor(dataset.shape[0]*args.train_prop)))]
    x_test = dataset[np.arange(int(np.floor(dataset.shape[0]*args.train_prop)), int(dataset.shape[0]))]

    del dataset  # Free memory
    nodes_history = {}
    train_length = int(x_train.shape[0]/n_nodes)
    train_sets = [x_train[i*train_length:(i + 1)*train_length] for i in range(0, n_nodes)]
    for i, train_set in enumerate(train_sets):
        print('--------------- Training node {} with {} epochs ---------------'.format(i, epochs[i]))
        node_history = {'ae': {}, 'svm': []}
        aae = AdversarialAutoencoder()
        train_loss, test_loss = aae.train(epochs=epochs[i], X_train=train_set, X_test=x_test, batch_size=args.batch_size)
        node_history['ae'] = {'train_loss': str(train_loss), 'test_loss': str(test_loss)}
        aae.save_model(i)
        aae.plot(i)
        print('\nCreating embeddings in node {}'.format(i))

        embeddings = aae.encoder.predict(train_set, batch_size=args.batch_size)
        embeddings = embeddings.reshape(embeddings.shape[0], 2048)

        test_embeddings = aae.encoder.predict(x_test, batch_size=args.batch_size)
        test_embeddings = test_embeddings.reshape(test_embeddings.shape[0], 2048)

        forged_embeddings = aae.encoder.predict(forged, batch_size=args.batch_size)
        forged_embeddings = forged_embeddings.reshape(forged_embeddings.shape[0], 2048)

        gammas = [0.5/2048]
        nus = [0.0001]

        for gamma in gammas:
            for nu in nus:
                classifier = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma, cache_size=2000, verbose=False)
                classifier.fit(embeddings)
                y_pred_train = classifier.predict(embeddings)
                y_pred_test = classifier.predict(test_embeddings)
                y_pred_outliers = classifier.predict(forged_embeddings)
                n_error_train = y_pred_train[y_pred_train == -1].size
                n_error_test = y_pred_test[y_pred_test == -1].size
                n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
                print('Gamma: {}, nu: {}'.format(gamma, nu))
                print('Error train: {}/{} --> {}%'.format(n_error_train, embeddings.shape[0],
                                                          100*n_error_train/embeddings.shape[0]))
                print('Error test: {}/{} --> {}%'.format(n_error_test, test_embeddings.shape[0],
                                                         100*n_error_test/test_embeddings.shape[0]))
                print('Error forged: {}/{} --> {}%'.format(n_error_outliers, forged_embeddings.shape[0],
                                                           100*n_error_outliers/forged_embeddings.shape[0]))

                true_positives = y_pred_outliers[y_pred_outliers == -1].size
                false_negatives = y_pred_outliers[y_pred_outliers == 1].size
                false_positives = y_pred_test[y_pred_test == -1].size
                true_negatives = y_pred_test[y_pred_test == 1].size

                precision = true_positives/(true_positives+false_positives)
                recall = true_positives/(true_positives+false_negatives)
                f1 = 2/(1/precision + 1/recall)
                print('Precision: {}, Recall: {}, F1: {}'.format(precision, recall, f1))
                node_history['svm'].append({str((gamma, nu)): {'precision': str(precision),
                                                               'recall': str(recall), 'f1': str(f1)}})
                pickle.dump(classifier, open('nodes/svm/{}_{}_{}.pkl'.format(i, gamma, nu), 'wb'))

    nodes_history[i] = node_history

    print(nodes_history)
    with open('nodes/history.json', 'w') as fp:
        json.dump(nodes_history, fp)
