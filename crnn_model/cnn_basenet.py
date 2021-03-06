"""The base convolution neural networks."""

from abc import ABCMeta

import numpy as np
import tensorflow as tf


class CNNBaseModel(metaclass=ABCMeta):
    """Base model for other specific cnn ctpn_models."""

    def __init__(self):
        """__init__ Constructor."""
        pass

    @staticmethod
    def conv2d(inputdata, out_channel, kernel_size, padding='SAME', stride=1,
               w_init=None, b_init=None, nl=tf.identity, split=1,
               use_bias=True, data_format='NHWC', name=None):
        """Pack the tensorflow conv2d function.

        Parameters
        ----------
        inputdata : numpy ndarray
            A 4D tensorflow tensor containing known number of channels,
            but can have other unknown dimensions..
        out_channel : int
            The number of output channel.
        kernel_size : int
            Only supports square kernel convolutions.
        padding : str
            'VALID' or 'SAME'.
        stride : int
            Only support square strides.
        w_init : int or None
            Initializer for convolution weights.
        b_init : int or None
            Initializer for bias.
        nl : function
            Returns a tensor with the same shape as input.
        split : int
            Split channels as used in Alexnet mainly group for
            GPU memory save.
        use_bias : bool
            Whether to use bias or not.
        data_format : str
            Default set to NHWC according tensorflow.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        tensor (numpy ndarray)
            tf.Tensor named ``output``.

        """
        with tf.variable_scope(name):
            in_shape = inputdata.get_shape().as_list()
            channel_axis = (3 if data_format == 'NHWC' else 1)
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, \
                '[Conv2D] Input cannot have unknown channel!'
            assert in_channel % split == 0
            assert out_channel % split == 0

            padding = padding.upper()

            if (isinstance(kernel_size, list)):
                filter_shape = [kernel_size[0], kernel_size[1]] \
                    + [in_channel / split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] \
                    + [in_channel / split, out_channel]

            if (isinstance(stride, list)):
                strides = ([1, stride[0], stride[1], 1] if data_format
                           == 'NHWC' else [1, 1, stride[0], stride[1]])
            else:
                strides = ([1, stride, stride, 1] if data_format
                           == 'NHWC' else [1, 1, stride, stride])

            if (w_init is None):
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if (b_init is None):
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if (use_bias):
                b = tf.get_variable('b', [out_channel],
                                    initializer=b_init)

            if (split == 1):
                conv = tf.nn.conv2d(inputdata, w, strides, padding,
                                    data_format=data_format)
            else:
                inputs = tf.split(inputdata, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding,
                           data_format=data_format) for (i, k) in
                           zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = nl((tf.nn.bias_add(conv, b,
                     data_format=data_format) if use_bias else conv),
                     name=name)

        return ret

    @staticmethod
    def relu(inputdata, name=None):
        """Compute Rectified Linear Unit.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        numpy ndarray
            An array containing the computed input data.

        """
        return tf.nn.relu(features=inputdata, name=name)

    @staticmethod
    def sigmoid(inputdata, name=None):
        """Compute sigmoid.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        numpy ndarray
            An array containing the output data.

        """
        return tf.nn.sigmoid(x=inputdata, name=name)

    @staticmethod
    def maxpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """Perform the max pooling on the input.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        kernel_size : int
            Only supports square kernel convolutions.
        stride : int
            Only support square strides.
        padding : str
            'VALID' or 'SAME'.
        data_format : str
            Default set to NHWC according tensorflow.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        type
            The max pooled output tensor.

        """
        padding = padding.upper()

        if (stride is None):
            stride = kernel_size

        if (isinstance(kernel_size, list)):
            kernel = ([1, kernel_size[0], kernel_size[1],
                      1] if (data_format == 'NHWC') else [1, 1,
                      kernel_size[0], kernel_size[1]])
        else:
            kernel = ([1, kernel_size, kernel_size, 1] if data_format
                      == 'NHWC' else [1, 1, kernel_size, kernel_size])

        if isinstance(stride, list):
            strides = ([1, stride[0], stride[1], 1] if (data_format
                       == 'NHWC') else [1, 1, stride[0], stride[1]])
        else:
            strides = \
                ([1, stride, stride, 1] if (data_format ==
                 'NHWC') else [1, 1, stride, stride])

        return tf.nn.max_pool(value=inputdata, ksize=kernel, strides=strides,
                              padding=padding, data_format=data_format,
                              name=name)

    @staticmethod
    def avgpooling(inputdata, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """Perform the average pooling on the input.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        kernel_size : int
            Only supports square kernel convolutions.
        stride : int
            Only support square strides.
        padding : str
            'VALID' or 'SAME'.
        data_format : str
            Default set to NHWC according tensorflow.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        tensor (numpy ndarray)
            The average pooled output tensor.

        """
        if (stride is None):
            stride = kernel_size

        kernel = ([1, kernel_size, kernel_size, 1] if (data_format
                  == 'NHWC') else [1, 1, kernel_size, kernel_size])

        strides = \
            ([1, stride, stride, 1] if (data_format ==
             'NHWC') else [1, 1, stride, stride])

        return tf.nn.avg_pool(value=inputdata, ksize=kernel, strides=strides,
                              padding=padding, data_format=data_format,
                              name=name)

    @staticmethod
    def globalavgpooling(inputdata, data_format='NHWC', name=None):
        """Compute the mean of elements across dimensions of a tensor.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        data_format : type
            Default set to NHWC according tensorflow.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        numpy ndarray
            The reduced tensor.

        """
        assert inputdata.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = ([1, 2] if data_format == 'NHWC' else [2, 3])

        return tf.reduce_mean(input_tensor=inputdata, axis=axis, name=name)

    @staticmethod
    def layernorm(inputdata, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """Normalize the input data.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        epsilon : float
            epsilon to avoid divide-by-zero.
        use_bias : bool
            whether to use the extra affine transformation or not.
        use_scale : bool
            whether to use the extra affine transformation or not.
        data_format : str
            Default set to NHWC according tensorflow.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        numpy ndarray
            The normalized, scaled, offset tensor.

        """
        shape = inputdata.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        (mean, var) = tf.nn.moments(inputdata, list(range(1,
                                    len(shape))), keep_dims=True)

        if (data_format == 'NCHW'):
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if (ndims == 2):
            new_shape = [1, channnel]

        if (use_bias):
            beta = tf.get_variable('beta', [channnel],
                                   initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if (use_scale):
            gamma = tf.get_variable('gamma', [channnel],
                                    initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma,
                                         epsilon, name=name)

    @staticmethod
    def instancenorm(inputdata, epsilon=1e-5, data_format='NHWC',
                     use_affine=True, name=None):
        """Normalize the input data.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        epsilon : float
            epsilon to avoid divide-by-zero.
        data_format : str
            Default set to NHWC according tensorflow.
        use_affine : bool
            Whether to use affine or not.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        numpy ndarray
            The normalized, scaled, offset tensor.

        """
        shape = inputdata.get_shape().as_list()
        if (len(shape) != 4):
            raise ValueError(
                'Input data of instancebn layer has to be 4D tensor'
                             )

        if (data_format == 'NHWC'):
            axis = [1, 2]
            ch = shape[3]
            new_shape = [1, 1, 1, ch]
        else:
            axis = [2, 3]
            ch = shape[1]
            new_shape = [1, ch, 1, 1]
        if (ch is None):
            raise ValueError('Input of instancebn require known channel!')

        (mean, var) = tf.nn.moments(inputdata, axis, keep_dims=True)

        if (not use_affine):
            return tf.divide(inputdata - mean, tf.sqrt(var + epsilon),
                             name='output')

        beta = tf.get_variable('beta', [ch],
                               initializer=tf.constant_initializer())
        beta = tf.reshape(beta, new_shape)
        gamma = tf.get_variable('gamma', [ch],
                                initializer=tf.constant_initializer(1.0))
        gamma = tf.reshape(gamma, new_shape)
        return tf.nn.batch_normalization(inputdata, mean, var, beta, gamma,
                                         epsilon, name=name)

    @staticmethod
    def dropout(inputdata, keep_prob, noise_shape=None, name=None):
        """Compute dropout.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        keep_prob : float
            (Deprecated) A deprecated alias for (1-rate).
        noise_shape : numpy 1-D array
            The shape for randomly generated keep/drop flags.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        numpy ndarray
            A Tensor of the same shape of the inputdata.

        """
        return tf.nn.dropout(inputdata, keep_prob=keep_prob,
                             noise_shape=noise_shape, name=name)

    @staticmethod
    def fullyconnect(inputdata, out_dim, w_init=None, b_init=None,
                     nl=tf.identity, use_bias=True, name=None):
        """Fully-Connect the input layers.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        out_dim : int
            output dimension.
        w_init : None or int
            initializer for w. Defaults to `variance_scaling_initializer`.
        b_init : None or int
            initializer for b. Defaults to zero.
        nl : function
            a nonlinearity function.
        use_bias : bool
            whether to use bias.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        tensor (numpy ndarray)
            2D Tensor named ``output`` with attribute `variables`.

        """
        shape = inputdata.get_shape().as_list()[1:]
        if (None not in shape):
            inputdata = tf.reshape(inputdata, [-1, int(np.prod(shape))])
        else:
            inputdata = tf.reshape(inputdata,
                                   tf.stack([tf.shape(inputdata)[0], -1]))

        if (w_init is None):
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if (b_init is None):
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=inputdata,
                              activation=lambda x: nl(x, name='output'),
                              use_bias=use_bias, name=name,
                              kernel_initializer=w_init,
                              bias_initializer=b_init, trainable=True,
                              units=out_dim)
        return ret

    @staticmethod
    def layerbn(inputdata, is_training):
        """Add a Batch Normalization layer.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containig the input data.
        is_training : bool
            Whether or not the layer is in training mode.

        Returns
        -------
        numpy ndarray
            A Tensor representing the output of the operation.

        """
        output = tf.contrib.layers.batch_norm(inputdata, scale=True,
                                              is_training=is_training,
                                              updates_collections=None)
        return output

    @staticmethod
    def squeeze(inputdata, axis=None, name=None):
        """Remove dimensions of size 1 from the shape of a tensor.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        axis : list
            An optional list of ints.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        type
            The same input tensor but with one or more removed dimensions
            of size 1.

        """
        return tf.squeeze(input=inputdata, axis=axis, name=name)
