import tensorflow as tf
import os
import numpy as np
import argparse
import sys
from sklearn import preprocessing, random_projection
from tqdm import tqdm
import cv2
from autoencoder import *
from classic_adversarial_autoencoder import *
from adversarial_autoencoder import *

MAX_IMAGES = 10000


def create_embeddings(model, name):
    short_saved = False
    for fol in ['forged_patches', 'pristine_patches']:
        print('Creating embeddings of {} of {}\n'.format(fol, name))
        embeddings = []
        test_imgs = os.listdir(os.path.join('data', fol))
        for i, idx in tqdm(enumerate(test_imgs)):

            im = cv2.imread(os.path.join('data', fol, idx))
            if im.shape != (64, 64, 3):
                continue

            im = (im.astype(np.float32)/255)  # Values 0 to 1
            encoding = model.encoder.predict(np.expand_dims(im, 0))
            encoding = encoding.reshape(2048, 1)
            embeddings.append(encoding)
            if i > MAX_IMAGES and fol == 'pristine_patches' and not short_saved:
                np.save('embeddings/short_{}_{}_emb.npy'.format(name, fol), np.squeeze(np.asarray(embeddings)))
                short_saved = True

        np.save('embeddings/{}_{}_emb.npy'.format(name, fol), np.squeeze(np.asarray(embeddings)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=bool, help='Create embeddings', default=False)
    parser.add_argument('--projector', type=str, help='ae, aae or c_aae, used to select which embeddings'
                                                      'will be projected, use 0 for not projecting', default='0')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args.embeddings:
        ae = Autoencoder()
        ae.autoencoder.load_weights('models/ae_autoencoder.h5')
        create_embeddings(ae, 'ae', True)

        aae = AdversarialAutoencoder()
        aae.autoencoder.load_weights('models/aae_autoencoder.h5')
        create_embeddings(aae, 'aae')

        c_aae = ClassicAdversarialAutoencoder()
        c_aae.autoencoder.load_weights('models/c_aae_autoencoder.h5')
        create_embeddings(c_aae, 'c_aae')

    if args.projector != '0':
        from tensorflow.contrib.tensorboard.plugins import projector

        pristine_emb = np.load('embeddings/short_{}_pristine_patches_emb.npy'.format(args.projector))
        forged_emb = np.load('embeddings/{}_forged_patches_emb.npy'.format(args.projector))

        embeddings = np.concatenate((pristine_emb, forged_emb))
        embeddings_var = tf.Variable(embeddings, name='embeddings')

        file = open('logs/metadata.tsv', 'a+')
        [file.write('0\n') for i in range(pristine_emb.shape[0])]
        [file.write('1\n') for i in range(forged_emb.shape[0])]
        file.close()

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            saver.save(session, 'logs/{}_model.ckpt'.format(args.projector), 1)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('logs')

        embedding = config.embeddings.add()
        embedding.tensor_name = embeddings_var.name
        embedding.metadata_path = 'metadata.tsv'

        projector.visualize_embeddings(summary_writer, config)
        os.system('tensorboard --logdir=logs/')





