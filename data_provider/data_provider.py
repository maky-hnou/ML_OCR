"""Provide the training and testing data for shadow net."""
import copy
import os

import cv2
import numpy as np

from .base_data_provider import Dataset


class TextDataset(Dataset):
    """Implement a dataset class providing the image and its corresponding text.

    Attributes
    ----------
    __normalization : str
        Where normalization is stored.
    __images : list
        Where images are stored.
    normalize_images : function
        Normalizes the images.
    __labels : numpy ndarray
        Where images labels are stored.
    __imagenames : list
        Where images names are stored.
    _epoch_images : list
        Where images epochs are stored.
    _epoch_labels : list
        Where labels epochs are stored.
    _epoch_imagenames : list
        Where image names epochs are stored.
    __shuffle : str
        Where shuffle is stored.
    shuffle_images_labels : function
        Shuffles image images, their labels and names.
    __batch_counter : int
        equal to 0.

    """

    def __init__(self, images, labels, imagenames, shuffle=None,
                 normalization=None):
        """Short summary.

        Parameters
        ----------
        images : list
            A list of ndarrays each one represents an image.
        labels : numpy ndarray
            A list of strings each one represents an image label.
        imagenames : list
            A list of strings each one represents an image name.
        shuffle : str
            A string representing whether to shuffle the data or not.
        normalization : str
            A string representing whether to normalize the data or not.

        Returns
        -------
        None

        """
        super(TextDataset, self).__init__()

        self.__normalization = normalization
        if (self.__normalization not in [None, 'divide_255', 'divide_256']):
            raise ValueError('normalization parameter wrong')
        self.__images = self.normalize_images(images, self.__normalization)

        self.__labels = labels
        self.__imagenames = imagenames
        self._epoch_images = copy.deepcopy(self.__images)
        self._epoch_labels = copy.deepcopy(self.__labels)
        self._epoch_imagenames = copy.deepcopy(self.__imagenames)

        self.__shuffle = shuffle
        if (self.__shuffle not in [None, 'once_prior_train', 'every_epoch']):
            raise ValueError('shuffle parameter wrong')
        if (self.__shuffle == 'every_epoch' or 'once_prior_train'):
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = \
                self.shuffle_images_labels(self._epoch_images,
                                           self._epoch_labels,
                                           self._epoch_imagenames)

        self.__batch_counter = 0

    @property
    def num_examples(self):
        """Check the number of images and the number of labels.

        Returns
        -------
        int
            The number of images labels.

        """
        assert self.__images.shape[0] == self.__labels.shape[0]
        return self.__labels.shape[0]

    @property
    def images(self):
        """Return the images per epoch.

        Returns
        -------
        numpy ndarray
            An array containg the images per epoch.

        """
        return self._epoch_images

    @property
    def labels(self):
        """Return the labels per epoch.

        Returns
        -------
        numpy ndarray
            A 2D array containig the images labels.

        """
        return self._epoch_labels

    @property
    def imagenames(self):
        """Return the images names per epoch.

        Returns
        -------
        numpy ndarray
            An array containing the images names per epoch.

        """
        return self._epoch_imagenames

    def next_batch(self, batch_size):
        """Return next batch size.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        int
            the batch size.

        """
        start = self.__batch_counter * batch_size
        end = (self.__batch_counter + 1) * batch_size
        self.__batch_counter += 1
        images_slice = self._epoch_images[start:end]
        labels_slice = self._epoch_labels[start:end]
        imagenames_slice = self._epoch_imagenames[start:end]
        # if overflow restart from the begining
        if (images_slice.shape[0] != batch_size):
            self.__start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice, imagenames_slice

    def __start_new_epoch(self):
        """Start new epoch and shuffle the images, their labels and names.

        Returns
        -------
        None

        """
        self.__batch_counter = 0

        if (self.__shuffle == 'every_epoch'):
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = \
                self.shuffle_images_labels(self._epoch_images,
                                           self._epoch_labels,
                                           self._epoch_imagenames)
        else:
            pass


class TextDataProvider(object):
    """Implement the text data provider for training and testing the shadow net.

    Attributes
    ----------
    __dataset_dir : str
        Where dataset_dir is stored.
    __validation_split : float or None
        Where validation_split is sotred.
    __shuffle : str
        Where shuffle is stored.
    __normalization : str
        Where normalization is stored.
    __train_dataset_dir : str
        Where train_dataset_dir is stored.
    __test_dataset_dir : str
        Where test_dataset_dir is stored.
    test : class
        Prepare the test data.
    train : class
        Prepare the train data.
    validation : class
        Prepare the validation data.

    """

    def __init__(self, dataset_dir, annotation_name, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None):
        """__init__ Constructor.

        Parameters
        ----------
        dataset_dir : str
            Where dataset is saved.
        annotation_name : str
            The name of the file containing the annotations.
        validation_set : numpy ndarra or None
            The validation set.
        validation_split : float or None
            float: chunk of `train set` will be marked as `validation set`.
            None: if 'validation set' == True, `validation set` will be
                  copy of `test set`.
        shuffle : str
            'once_prior_train': represent shuffle only once before training
            'every_epoch': represent shuffle the data every epoch
        normalization : str
            'None': no any normalization
            'divide_255': divide all pixels by 255
            'divide_256': divide all pixels by 256
            'by_chanels': substract mean of every chanel
                          and divide each chanel data by
                          it's standart deviation

        Returns
        -------
        None

        """
        self.__dataset_dir = dataset_dir
        self.__validation_split = validation_split
        self.__shuffle = shuffle
        self.__normalization = normalization
        self.__train_dataset_dir = os.path.join(self.__dataset_dir, 'Train')
        self.__test_dataset_dir = os.path.join(self.__dataset_dir, 'Test')
        assert os.path.exists(dataset_dir)
        assert os.path.exists(self.__train_dataset_dir)
        assert os.path.exists(self.__test_dataset_dir)

        # add test dataset
        test_anno_path = os.path.join(self.__test_dataset_dir, annotation_name)
        assert os.path.exists(test_anno_path)

        with open(test_anno_path, 'r') as anno_file:
            info = np.array([tmp.strip().split() for tmp in
                            anno_file.readlines()])

            test_images_org = [cv2.imread(os.path.join(self.__test_dataset_dir,
                               tmp), cv2.IMREAD_COLOR)
                               for tmp in info[:, 0]]
            test_images = np.array([cv2.resize(tmp, (100, 32)) for tmp in
                                    test_images_org])

            test_labels = np.array([tmp for tmp in info[:, 1]])

            test_imagenames = np.array([os.path.basename(tmp) for tmp in
                                        info[:, 0]])

            self.test = TextDataset(test_images, test_labels,
                                    imagenames=test_imagenames,
                                    shuffle=shuffle,
                                    normalization=normalization)

        anno_file.close()

        # add train and validation dataset
        train_anno_path = os.path.join(self.__train_dataset_dir,
                                       annotation_name)
        assert os.path.exists(train_anno_path)

        with open(train_anno_path, 'r') as anno_file:
            info = np.array([tmp.strip().split() for tmp in
                             anno_file.readlines()])

            train_images_org = \
                [cv2.imread(os.path.join(self.__train_dataset_dir, tmp),
                            cv2.IMREAD_COLOR) for tmp in info[:, 0]]
            train_images = np.array([cv2.resize(tmp, (100, 32)) for tmp in
                                     train_images_org])

            train_labels = np.array([tmp for tmp in info[:, 1]])
            train_imagenames = np.array([os.path.basename(tmp) for tmp in
                                         info[:, 0]])

            if (validation_set is not None and validation_split is not None):
                split_idx = int(train_images.shape[0] * (1 - validation_split))
                self.train = TextDataset(
                                images=train_images[:split_idx],
                                labels=train_labels[:split_idx],
                                shuffle=shuffle,
                                normalization=normalization,
                                imagenames=train_imagenames[:split_idx]
                                )
                self.validation = TextDataset(
                                    images=train_images[split_idx:],
                                    labels=train_labels[split_idx:],
                                    shuffle=shuffle,
                                    normalization=normalization,
                                    imagenames=train_imagenames[split_idx:]
                                    )
            else:
                self.train = TextDataset(images=train_images,
                                         labels=train_labels,
                                         shuffle=shuffle,
                                         normalization=normalization,
                                         imagenames=train_imagenames)

            if (validation_set and not validation_split):
                self.validation = self.test
        anno_file.close()
        return

    def __str__(self):
        """Return formatted information about data.

        Returns
        -------
        string
            A string containing info about the prepared data.

        """
        provider_info = 'Dataset_dir: {:s} contain training images: {:d} valid'
        'ation images: {:d} testing '
        'images: {:d}'.format(self.__dataset_dir, self.train.num_examples,
                              self.validation.num_examples,
                              self.test.num_examples)
        return provider_info

    @property
    def dataset_dir(self):
        """Return the dataset dir.

        Returns
        -------
        string
            A string containing the dataset dir.

        """
        return self.__dataset_dir

    @property
    def train_dataset_dir(self):
        """Return the train dataset dir.

        Returns
        -------
        string
            A string containing the  train dataset dir.

        """
        return self.__train_dataset_dir

    @property
    def test_dataset_dir(self):
        """Resturn the test dataset dir.

        Returns
        -------
        string
            A string containing the test dataset dir.

        """
        return self.__test_dataset_dir
