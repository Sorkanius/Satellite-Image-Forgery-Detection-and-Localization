from sklearn.svm import OneClassSVM
from sklearn import random_projection
from sklearn import preprocessing
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
    data_length = int(pristine_emb.shape[0])
    print('Data Loaded, {} pristine vectors'.format(data_length))

    np.random.seed(7)
    idx = np.arange(0, data_length)
    np.random.shuffle(idx)

    X_train = pristine_emb[idx[:int(args.train_prop*data_length)]]
    X_test = pristine_emb[idx[int(args.train_prop*data_length):]]

    # scaler = preprocessing.StandardScaler().fit(X_train)
    # scaler = random_projection.GaussianRandomProjection(n_components=1024).fit(X_train)
    # scaler = preprocessing.MinMaxScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # forged_emb = scaler.transform(forged_emb)

    gammas = [1/2048, 2/2048, 0.5/2048]
    nus = [0.00001, 0.0001, 0.000001]

    print('Starting training on {} train vectors with {} test vectors'.format(X_train.shape[0], X_test.shape[0]))

    for gamma in gammas:
        for nu in nus:
            classifier = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma, cache_size=2000, verbose=False)
            classifier.fit(X_train)

            y_pred_train = classifier.predict(X_train)
            y_pred_test = classifier.predict(X_test)
            y_pred_outliers = classifier.predict(forged_emb)
            n_error_train = y_pred_train[y_pred_train == -1].size
            n_error_test = y_pred_test[y_pred_test == -1].size
            n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
            print('Gamma: {}, nu: {}'.format(gamma, nu))
            print('Error train: {}/{} --> {}%'.format(n_error_train, X_train.shape[0],
                                                      100*n_error_train/X_train.shape[0]))
            print('Error test: {}/{} --> {}%'.format(n_error_test, X_test.shape[0], 100*n_error_test/X_test.shape[0]))
            print('Error forged: {}/{} --> {}%'.format(n_error_outliers, forged_emb.shape[0],
                                                       100*n_error_outliers/forged_emb.shape[0]))

            true_positives = y_pred_outliers[y_pred_outliers == -1].size
            false_negatives = y_pred_outliers[y_pred_outliers == 1].size
            false_positives = y_pred_test[y_pred_test == -1].size
            true_negatives = y_pred_test[y_pred_test == 1].size

            precision = true_positives/(true_positives+false_positives)
            recall = true_positives/(true_positives+false_negatives)
            f1 = 2/(1/precision + 1/recall)
            print('Precision: {}, Recall: {}, F1: {}'.format(precision, recall, f1))
