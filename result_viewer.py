import argparse
import pickle
import sys
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=bool, help='Save of images or just show', default=True)
    parser.add_argument('--analysis', type=bool, help='Analyze the values of each loss', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    models_dir = 'models/'

    with open(models_dir + 'c_aae_history.pkl', 'rb') as input_file:
        c_aae_history = pickle.load(input_file)

    with open(models_dir + 'aae_history.pkl', 'rb') as input_file:
        aae_history = pickle.load(input_file)

    with open(models_dir + 'ae_history.pkl', 'rb') as input_file:
        ae_history = pickle.load(input_file)

    plt.figure()
    plt.title('Loss History')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    step = len(c_aae_history['rec_loss']) // 10 if len(c_aae_history['rec_loss']) > 100000 else 1
    plt.plot(np.arange(len(c_aae_history['rec_loss'][::step])), c_aae_history['rec_loss'][::step],
             c='C0', label='Classic AAE')
    plt.plot(np.arange(len(aae_history['g_loss'][::step])), aae_history['g_loss'][::step],
             c='C1', label='AAE')
    plt.plot(np.arange(len(ae_history['ae_loss'][::step])), ae_history['ae_loss'][::step],
             c='C2', label='AE')
    plt.legend()
    plt.grid()

    if args.save:
        plt.savefig('results/loss')
    else:
        plt.show()

    plt.figure()
    plt.title('Accuracy History')
    plt.xlabel('Iter')
    plt.ylabel('Acc')
    step = len(c_aae_history['reg_acc']) // 10 if len(c_aae_history['reg_acc']) > 100000 else 1
    plt.plot(np.arange(len(c_aae_history['reg_acc'][::step])), c_aae_history['reg_acc'][::step],
             c='C0', label='Discriminator - Classic AAE')
    plt.plot(np.arange(len(aae_history['d_acc'][::step])), aae_history['d_acc'][::step],
             c='C1', label='Discriminator - AAE')
    plt.legend()
    plt.grid()

    if args.save:
        plt.savefig('results/accuracy')
    else:
        plt.show()

    plt.figure()
    plt.title('Loss History')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    step = len(c_aae_history['rec_test_loss']) // 10 if len(c_aae_history['rec_test_loss']) > 100000 else 1
    plt.plot(np.arange(len(c_aae_history['rec_test_loss'][::step])), c_aae_history['rec_test_loss'][::step],
             c='C0', label='Classic AAE')
    plt.plot(np.arange(len(aae_history['g_test_loss'][::step])), aae_history['g_test_loss'][::step],
             c='C1', label='AAE')
    plt.plot(np.arange(len(ae_history['ae_test_loss'][::step])), ae_history['ae_test_loss'][::step],
             c='C2', label='AE')
    plt.legend()
    plt.grid()

    if args.save:
        plt.savefig('results/test_loss')
    else:
        plt.show()

    plt.figure()
    plt.title('Accuracy History')
    plt.xlabel('Iter')
    plt.ylabel('Acc')
    step = len(c_aae_history['reg_test_acc']) // 10 if len(c_aae_history['reg_test_acc']) > 100000 else 1
    plt.plot(np.arange(len(c_aae_history['reg_test_acc'][::step])), c_aae_history['reg_test_acc'][::step],
             c='C0', label='Discriminator - Classic AAE')
    plt.plot(np.arange(len(aae_history['d_test_acc'][::step])), aae_history['d_test_acc'][::step],
             c='C1', label='Discriminator - AAE')
    plt.legend()
    plt.grid()

    if args.save:
        plt.savefig('results/test_accuracy')
    else:
        plt.show()

    if args.analysis:
        ae_test_loss = ae_history['ae_test_loss']
        aae_test_loss = aae_history['g_test_loss']
        c_aae_loss = c_aae_history['rec_test_loss']
        iterations = len(ae_test_loss)
        print('Mean loss in last 25% of iterations: ')
        print('AE: {}'.format(np.mean(ae_test_loss[int(0.75*iterations):])))
        print('AAE: {}'.format(np.mean(aae_test_loss[int(0.75*iterations):])))
        print('C_AAE: {}'.format(np.mean(c_aae_loss[int(0.75*iterations):])))

        print('\nMean loss in last 10% of iterations: ')
        print('AE: {}'.format(np.mean(ae_test_loss[int(0.90*iterations):])))
        print('AAE: {}'.format(np.mean(aae_test_loss[int(0.90*iterations):])))
        print('C_AAE: {}'.format(np.mean(c_aae_loss[int(0.90*iterations):])))

        print('\nMean loss in last 5% of iterations: ')
        print('AE: {}'.format(np.mean(ae_test_loss[int(0.95*iterations):])))
        print('AAE: {}'.format(np.mean(aae_test_loss[int(0.95*iterations):])))
        print('C_AAE: {}'.format(np.mean(c_aae_loss[int(0.95*iterations):])))

        print('\nLowest loss: ')
        print('AE: {}'.format(min(ae_test_loss)))
        print('AAE: {}'.format(min(aae_test_loss)))
        print('C_AAE: {}'.format(min(c_aae_loss)))
