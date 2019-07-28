"""Generate tfrecords."""

import os

from tools.write_text_features import write_features

if (__name__ == '__main__'):
    # Check if the dataset dir exists or not.
    if (not os.path.exists('./imgs/')):
        raise ValueError('Dataset {:s} doesn\'t exist'.format('./imgs/'))

    # Write tf records
    write_features(imgs_dir='./imgs/', tfrec_dir='./tfrec/')
