"""Train the RCNN."""

import os

from tools.train_shadownet import train_shadownet

if __name__ == '__main__':
    train_dir = './tfrec/'
    if (not os.path.exists(train_dir)):
        os.makedirs(train_dir)

    train_shadownet(dataset_dir=train_dir, weights_path=None)
    print('Done')
