from sklearn.svm import OneClassSVM
import pickle
import numpy as np
import sys
import argparse


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='ae, aae or c_aae, used to select which embeddings'
                                                  'will the svm use', required=True)
    parser.add_argument('--train_prop', type=float, help='Pristine data train proportion', default=0.8)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    pristine_emb = np.load('embeddings/short_{}_pristine_patches_emb.npy'.format(args.model))
    forged_emb = np.load('embeddings/{}_forged_patches_emb.npy'.format(args.model))
    data_length = int(pristine_emb.shape[0]*1)
    print('Data Loaded, {} pristine vectors'.format(data_length))

    idx = np.arange(0, data_length)
    np.random.shuffle(idx)

    X_train = pristine_emb[idx[:int(args.train_prop*data_length)]]
    X_test = pristine_emb[idx[int(args.train_prop*data_length):]]

    print('Starting training on {} train vectors with {} test vectors'.format(X_train.shape[0], X_test.shape[0]))
    classifier = OneClassSVM(nu=0.00001, kernel='rbf', gamma=8/2048, cache_size=1000, verbose=True)
    classifier.fit(X_train)
    print('Finishing training')

    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    y_pred_outliers = classifier.predict(forged_emb)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    print('Error train: {}/{} --> {}%'.format(n_error_train, X_train.shape[0], 100*n_error_train/X_train.shape[0]))
    print('Error test: {}/{} --> {}%'.format(n_error_test, X_test.shape[0], 100*n_error_test/X_test.shape[0]))
    print('Error forged: {}/{} --> {}%'.format(n_error_outliers, forged_emb.shape[0],
                                               100*n_error_outliers/forged_emb.shape[0]))

    pickle.dump(classifier, open('models/{}_svm.pkl'.format(args.model), 'wb'))

