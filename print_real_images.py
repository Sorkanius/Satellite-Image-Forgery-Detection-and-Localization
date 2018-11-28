import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def sample_images(imgs):
        
        r, c = 5, 5

        imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('real.png')
        plt.close()


if __name__ == '__main__':
    X_train = np.load('data.npy')
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train.astype(np.float32) - mean) / (std + 1e-7)
    idx = np.arange(0, 25)
    sample_images(X_train[idx])

