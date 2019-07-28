"""Write text features into tensorflow records."""
import os

import numpy as np
from data_provider.data_provider import TextDataProvider
from local_utils.data_utils import TextFeatureIO


def write_features(imgs_dir, tfrec_dir):
    """Generate TFrecords from images.

    Parameters
    ----------
    imgs_dir : str
        The directory containing the images.
    tfrec_dir : str
        The directory where TFrecords will be saved.

    Returns
    -------
    None

    """
    if (not os.path.exists(tfrec_dir)):
        os.makedirs(tfrec_dir)

    print('Initialize the dataset provider ......')
    provider = TextDataProvider(
        dataset_dir=imgs_dir,
        annotation_name='sample.txt',
        validation_set=True,
        validation_split=0.15,
        shuffle='every_epoch',
        normalization=None,
        )
    print('Dataset provider intialize complete')

    feature_io = TextFeatureIO()

    # write train tfrecords
    print('Start writing training tf records')

    train_images = provider.train.images
    train_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in
                    train_images]
    train_labels = provider.train.labels
    train_imagenames = provider.train.imagenames

    train_tfrecord_path = os.path.join(tfrec_dir, 'train_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=train_tfrecord_path,
                                     labels=train_labels,
                                     images=train_images,
                                     imagenames=train_imagenames)

    # write test tfrecords
    print('Start writing testing tf records')

    test_images = provider.test.images
    test_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in
                   test_images]
    test_labels = provider.test.labels
    test_imagenames = provider.test.imagenames

    test_tfrecord_path = os.path.join(tfrec_dir, 'test_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=test_tfrecord_path,
                                     labels=test_labels,
                                     images=test_images,
                                     imagenames=test_imagenames)

    # write val tfrecords
    print('Start writing validation tf records')

    val_images = provider.validation.images
    val_images = [bytes(list(np.reshape(tmp, [100 * 32 * 3]))) for tmp in
                  val_images]
    val_labels = provider.validation.labels
    val_imagenames = provider.validation.imagenames

    val_tfrecord_path = os.path.join(tfrec_dir, 'validation_feature.tfrecords')
    feature_io.writer.write_features(tfrecords_path=val_tfrecord_path,
                                     labels=val_labels,
                                     images=val_images,
                                     imagenames=val_imagenames)
