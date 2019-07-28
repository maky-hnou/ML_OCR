"""Implement some utils.

The utils are used to convert image and it's corresponding
label into tfrecords
"""
import os
import sys

import numpy as np
import tensorflow as tf

from .establish_char_dict import CharDictBuilder


class FeatureIO(object):
    """Implement the base writer class.

    Attributes
    ----------
    __char_list : list
        A list containing the char_dict values.
    __ord_map : list
        A list containing the ord_map_dict values.

    """

    def __init__(self, char_dict_path='data/char_dict/char_dict.json',
                 ord_map_dict_path='data/char_dict/ord_map.json'):
        """__init__ constructor.

        Parameters
        ----------
        char_dict_path : str
            A strng containing the char_dict_path.
        ord_map_dict_path : str
            A string containing the ord_map_dict_path.

        Returns
        -------
        None

        """
        self.__char_list = \
            CharDictBuilder.read_char_dict(char_dict_path)
        self.__ord_map = \
            CharDictBuilder.read_ord_map_dict(
                ord_map_dict_path
                )
        return

    @property
    def char_list(self):
        """Return char_list.

        Returns
        -------
        list
            A list of strings representing the char_list.

        """
        return self.__char_list

    @staticmethod
    def int64_feature(value):
        """Insert int64 features into Example proto.

        Parameters
        ----------
        value : list
            A list of integers.

        Returns
        -------
        list
            Wrapped list of int64.

        """
        if (not isinstance(value, list)):
            value = [value]
        value_tmp = []
        is_int = True
        for val in value:
            if (not isinstance(val, int)):
                is_int = False
                value_tmp.append(int(float(val)))
        if (is_int is False):
            value = value_tmp
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        """Insert float features into Example proto.

        Parameters
        ----------
        value : list
            A list of floats.

        Returns
        -------
        list
            Wrapped list of floats.

        """
        if (not isinstance(value, list)):
            value = [value]
        value_tmp = []
        is_float = True
        for val in value:
            if (not isinstance(val, int)):
                is_float = False
                value_tmp.append(float(val))
        if (is_float is False):
            value = value_tmp
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        """Insert bytes features into Example proto.

        Parameters
        ----------
        value : list
            A list of bytes.

        Returns
        -------
        list
            Wrapped list of bytes.

        """
        if (not isinstance(value, bytes)):
            if (not isinstance(value, list)):
                value = value.encode('utf-8')
            else:
                value = [val.encode('utf-8') for val in value]
        if (not isinstance(value, list)):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def char_to_int(self, char):
        """Return the ord of a char.

        Parameters
        ----------
        char : str
            A string containing a char.

        Returns
        -------
        int
            The ord of a char.

        """
        temp = ord(char)
        # convert upper character into lower character
        if (65 <= temp <= 90):
            temp = temp + 32

        for k, v in self.__ord_map.items():
            if (v == str(temp)):
                temp = int(k)
                return temp
        raise KeyError("Character {} missing in ord_map.json".format(char))

    def int_to_char(self, number):
        """Return the corresponding char to the input int from char_list.

        Parameters
        ----------
        number : int
            Description of parameter `number`.

        Returns
        -------
        string
            The corresponding char to the input int.

        """
        if (number == '1'):
            return '*'
        if (number == 1):
            return '*'
        else:
            return self.__char_list[str(number)]

    def encode_labels(self, labels):
        """Encode the labels for ctc loss.

        Parameters
        ----------
        labels : numpy ndarray
            An array containing the images labels.

        Returns
        -------
        numpy ndarray
            Arrays containing the encoded labels and their lengths.

        """
        encoded_labeles = []
        lengths = []
        for label in labels:
            encode_label = [self.char_to_int(char) for char in label]
            encoded_labeles.append(encode_label)
            lengths.append(len(label))
        return encoded_labeles, lengths

    def sparse_tensor_to_str(self, spares_tensor: tf.SparseTensor):
        """Sparse tensor to string.

        Parameters
        ----------
        spares_tensor : tf.SparseTensor
            A tensorflow tensor.

        Returns
        -------
        string
            A tensor parsed to astring.

        """
        indices = spares_tensor.indices
        values = spares_tensor.values
        values = np.array([self.__ord_map[str(tmp)] for tmp in values])
        dense_shape = spares_tensor.dense_shape

        number_lists = np.ones(dense_shape, dtype=values.dtype)
        str_lists = []
        res = []
        for i, index in enumerate(indices):
            number_lists[index[0], index[1]] = values[i]
        for number_list in number_lists:
            str_lists.append([self.int_to_char(val) for val in number_list])
        for str_list in str_lists:
            res.append(''.join(c for c in str_list if (c != '*')))
        return res

    def str_to_sparse_tensor(self, texts):
        """Sparse strings to tensor.

        Parameters
        ----------
        texts : str
            Description of parameter `texts`.

        Returns
        -------
        lists
            Strings appended to lists.

        """
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.char_to_int(c) for c in text]
            # sparse tensor must have size of max. label-string
            if (len(labelStr) > shape[1]):
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)


class TextFeatureWriter(FeatureIO):
    """Implement the CRNN feature writer."""

    def __init__(self):
        """__init__ constructor.

        Returns
        -------
        None

        """
        super(TextFeatureWriter, self).__init__()
        return

    def write_features(self, tfrecords_path, labels, images, imagenames):
        """Write images, labels, images names into tfrecords.

        Parameters
        ----------
        tfrecords_path : str
            Where tfrecords will be saved.
        labels : numpy ndarray
            An array containing the images labels and their indexes.
        images : numpy ndarray
            An array containing the images and their indexes.
        imagenames : numpy ndarray
            An array containing the images names and their indexes.

        Returns
        -------
        None

        """
        assert len(labels) == len(images) == len(imagenames)

        labels, length = self.encode_labels(labels)

        if (not os.path.exists(os.path.split(tfrecords_path)[0])):
            os.makedirs(os.path.split(tfrecords_path)[0])

        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for index, image in enumerate(images):
                features = tf.train.Features(feature={
                    'labels': self.int64_feature(labels[index]),
                    'images': self.bytes_feature(image),
                    'imagenames': self.bytes_feature(imagenames[index])
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
                sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecor'
                                 'ds'.format(index+1, len(images),
                                             imagenames[index]))
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
        return


class TextFeatureReader(FeatureIO):
    """Implement the crnn feature reader."""

    def __init__(self):
        """__init__ constructor.

        Returns
        -------
        None

        """
        super(TextFeatureReader, self).__init__()
        return

    @staticmethod
    def read_features(tfrecords_path, num_epochs):
        """Read tfrecords.

        Parameters
        ----------
        tfrecords_path : str
            Where tfrecords are saved.
        num_epochs : int
            The number of epochs.

        Returns
        -------
        numpy ndarrays
            Arrays containing the images, their names and labels.

        """
        assert os.path.exists(tfrecords_path)

        filename_queue = tf.train.string_input_producer([tfrecords_path],
                                                        num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = \
            tf.parse_single_example(
                serialized_example,
                features={
                          'images': tf.FixedLenFeature((), tf.string),
                          'imagenames': tf.FixedLenFeature([1], tf.string),
                          'labels': tf.VarLenFeature(tf.int64),
                          }
                )
        image = tf.decode_raw(features['images'], tf.uint8)
        images = tf.reshape(image, [32, 100, 3])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
        imagenames = features['imagenames']
        return images, labels, imagenames


class TextFeatureIO(object):
    """Implement a crnn feture io manager.

    Attributes
    ----------
    __writer : None
        Write images, labels names into tfrecords.
    __reader : numpy ndarrays
        Arrays containing the images, their names and labels.

    """

    def __init__(self):
        """__init__ constructor.

        Returns
        -------
        None

        """
        self.__writer = TextFeatureWriter()
        self.__reader = TextFeatureReader()
        return

    @property
    def writer(self):
        """Return self.__writer.

        Returns
        -------
        None
            Write tfrecords.

        """
        return self.__writer

    @property
    def reader(self):
        """Read tfrecords.

        Returns
        -------
        numpy ndarrays
            Arrays containing images, labels and names.

        """
        return self.__reader
