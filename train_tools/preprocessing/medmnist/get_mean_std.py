import numpy as np

def report_mean_std_3d(train_file='/media/zoe/ssd/dermamnist/train_images.npy'):
    train = np.load(train_file)
    mean, std = [], []
    imgs = train / 255.
    for channel in range(3):
        ch = imgs[:, :, :, channel].ravel()
        mean.append(np.mean(ch))
        std.append(np.std(ch))
    print('Mean: ', mean)
    print('Std: ', std)
    return mean, std

report_mean_std_3d()
report_mean_std_3d('/media/zoe/ssd/bloodmnist/train_images.npy')