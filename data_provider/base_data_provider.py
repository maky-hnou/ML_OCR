"""Base class for data provider."""
import numpy as np


class Dataset(object):
    """Implement some global useful functions used in all dataset."""

    def __init__(self):
        """__init__ Constructor."""
        pass

    @staticmethod
    def shuffle_images_labels(images, labels, imagenames):
        """Shuffle the images labels.

        Parameters
        ----------
        images : list
            A list of ndarrays each one represents an image.
        labels : list
            A list of strings each one represents an image label.
        imagenames : list
            A list of strings each one represent an image name .

        Returns
        -------
        lists
            Lists of the same input but shuffled.

        """
        images = np.array(images)
        labels = np.array(labels)

        assert images.shape[0] == labels.shape[0]

        random_index = np.random.permutation(images.shape[0])
        shuffled_images = images[random_index]
        shuffled_labels = labels[random_index]
        shuffled_imagenames = imagenames[random_index]

        return shuffled_images, shuffled_labels, shuffled_imagenames

    @staticmethod
    def normalize_images(images, normalization_type):
        """Normalize the images.

        Parameters
        ----------
        images : numpy ndarray
            A 4D array of images.
        normalization_type : str
            A string containing the value that the images will be normalized
            by(255, 255, by_chanels).

        Returns
        -------
        numpy ndarray
            An array containing the normalized images.

        """
        if (normalization_type == 'divide_255'):
            images = images / 255
        elif normalization_type == 'divide_256':
            images = images / 256
        elif normalization_type is None:
            pass
        else:
            raise Exception("Unknown type of normalization")
        return images

    def normalize_all_images_by_channels(self, initial_images):
        """Normalize the images by channels number.

        Parameters
        ----------
        initial_images : numpy ndarray
            A 4D array representing the images.

        Returns
        -------
        numpy ndarray
            An array containig the normalized images by channels number.

        """
        new_images = np.zeros(initial_images.shape)
        for i in range(initial_images.shape[0]):
            new_images[i] = self.normalize_image_by_channel(initial_images[i])
        return new_images

    @staticmethod
    def normalize_image_by_channel(image):
        """Normalize an image by channels number.

        Parameters
        ----------
        image : numpy ndarray
            An array representing the image.

        Returns
        -------
        numpy ndarray
            An array representing the normalized image.

        """
        new_image = np.zeros(image.shape)
        for chanel in range(3):
            mean = np.mean(image[:, :, chanel])
            std = np.std(image[:, :, chanel])
            new_image[:, :, chanel] = (image[:, :, chanel] - mean) / std
        return new_image

    def num_examples(self):
        """Raise NotImplementedError exception.

        Returns
        -------
        BaseException
            Raise NotImplementedError exception.

        """
        raise NotImplementedError

    def next_batch(self, batch_size):
        """Check the batch size.

        Parameters
        ----------
        batch_size : int
            The batch size.

        Returns
        -------
        BaseException
            Raise NotImplementedError exception.

        """
        raise NotImplementedError
