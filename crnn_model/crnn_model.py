"""Implement the CRNN model for squence recognition.

Implement the CRNN model in An End-to-End Trainable Neural Network for
Image-based Sequence Recognition and Its Application to Scene Text Recognition
paper
"""

import tensorflow as tf
from crnn_model import cnn_basenet
from tensorflow.contrib import rnn


class ShadowNet(cnn_basenet.CNNBaseModel):
    """Implement the crnn model for squence recognition.

    Attributes
    ----------
    __phase : str
        Where phase is stored.
    __hidden_nums : int
        Where hidden_nums is stored.
    __layers_nums : int
        Where layers_nums is stored.
    __seq_length : int
        Where seq_length is stored.
    __num_classes : int
        Where num_classes is stored.

    """

    def __init__(self, phase, hidden_nums,
                 layers_nums, seq_length, num_classes):
        """__init__ Constructor.

        Parameters
        ----------
        phase : str
            Whether training or testing the CRNN model.
        hidden_nums : int
            The number of grayscale.
        layers_nums : int
            The number of layers.
        seq_length : int
            The max length of the characters sequence.
        num_classes : int
            The number of classes (characters num + 1 for the space character).

        Returns
        -------
        None

        """
        super(ShadowNet, self).__init__()
        self.__phase = phase
        self.__hidden_nums = hidden_nums
        self.__layers_nums = layers_nums
        self.__seq_length = seq_length
        self.__num_classes = num_classes
        return

    @property
    def phase(self):
        """Return the phase.

        Returns
        -------
        str
            A string containg the current phase (Train or Test).

        """
        return self.__phase

    @phase.setter
    def phase(self, value):
        """Check the value of the phase variable.

        Parameters
        ----------
        value : str
            A string containing the phase value.

        Returns
        -------
        None

        """
        if (not isinstance(value, str)):
            raise TypeError('value should be a str \'Test\' or \'Train\'')
        if (value.lower() not in ['test', 'train']):
            raise ValueError('value should be a str \'Test\' or \'Train\'')
        self.__phase = value.lower()
        return

    def __conv_stage(self, inputdata, out_dims, name=None):
        """Traditional conv stage in VGG format.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.
        out_dims : int
            The dimension of the output tensor.
        name : None or str
            The name of the operation (optional).

        Returns
        -------
        numpy ndarray
            The input data max_pooled.

        """
        conv = self.conv2d(inputdata=inputdata, out_channel=out_dims,
                           kernel_size=3, stride=1, use_bias=False, name=name)
        relu = self.relu(inputdata=conv)
        max_pool = self.maxpooling(inputdata=relu, kernel_size=2, stride=2)
        return max_pool

    def __feature_sequence_extraction(self, inputdata):
        """Implement the 2.1 Part Feature Sequence Extraction.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data(batch*32*100*3 NHWC format).

        Returns
        -------
        numpy ndarray
            An array containing the output data.

        """
        # batch*16*50*64
        conv1 = self.__conv_stage(inputdata=inputdata, out_dims=64,
                                  name='conv1')
        # batch*8*25*128
        conv2 = self.__conv_stage(inputdata=conv1, out_dims=128,
                                  name='conv2')
        # batch*8*25*256
        conv3 = self.conv2d(inputdata=conv2, out_channel=256, kernel_size=3,
                            stride=1, use_bias=False, name='conv3')
        # batch*8*25*256
        relu3 = self.relu(conv3)
        # batch*8*25*256
        conv4 = self.conv2d(inputdata=relu3, out_channel=256, kernel_size=3,
                            stride=1, use_bias=False, name='conv4')
        # batch*8*25*256
        relu4 = self.relu(conv4)
        # batch*4*25*256
        max_pool4 = self.maxpooling(inputdata=relu4, kernel_size=[2, 1],
                                    stride=[2, 1], padding='VALID')
        # batch*4*25*512
        conv5 = self.conv2d(inputdata=max_pool4, out_channel=512,
                            kernel_size=3, stride=1, use_bias=False,
                            name='conv5')
        # batch*4*25*512
        relu5 = self.relu(conv5)
        if (self.phase.lower() == 'train'):
            bn5 = self.layerbn(inputdata=relu5, is_training=True)
        else:
            # batch*4*25*512
            bn5 = self.layerbn(inputdata=relu5, is_training=False)
        # batch*4*25*512
        conv6 = self.conv2d(inputdata=bn5, out_channel=512, kernel_size=3,
                            stride=1, use_bias=False, name='conv6')
        # batch*4*25*512
        relu6 = self.relu(conv6)
        if (self.phase.lower() == 'train'):
            bn6 = self.layerbn(inputdata=relu6, is_training=True)
        else:
            # batch*4*25*512
            bn6 = self.layerbn(inputdata=relu6, is_training=False)
        # batch*2*25*512
        max_pool6 = self.maxpooling(inputdata=bn6, kernel_size=[2, 1],
                                    stride=[2, 1])
        # batch*1*25*512
        conv7 = self.conv2d(inputdata=max_pool6, out_channel=512,
                            kernel_size=2, stride=[2, 1],
                            use_bias=False, name='conv7')
        # batch*1*25*512
        relu7 = self.relu(conv7)
        return relu7

    def __map_to_sequence(self, inputdata):
        """Implement the map to sequence.

        Implement the map to sequence part of the network mainly used to
         convert the cnn feature map to sequence used in
         later stacked lstm layers
        Parameters
        ----------
        inputdata : numpy ndarray
            An array containg the input data.

        Returns
        -------
        numpy ndarray
            An array containing the output data.

        """
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return self.squeeze(inputdata=inputdata, axis=1)

    def __sequence_label(self, inputdata):
        """Implement the sequence label part of the network.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.

        Returns
        -------
        None

        """
        with tf.variable_scope('LSTMLayers'):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in
                            [self.__hidden_nums, self.__hidden_nums]]
            # Backward direction cells
            bw_cell_list = [rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in
                            [self.__hidden_nums, self.__hidden_nums]]

            (stack_lstm_layer, _, _) = \
                rnn.stack_bidirectional_dynamic_rnn(
                    fw_cell_list, bw_cell_list,
                    inputdata, dtype=tf.float32
                    )

            if (self.phase.lower() == 'train'):
                stack_lstm_layer = self.dropout(inputdata=stack_lstm_layer,
                                                keep_prob=0.5)

            # [batch, width, 2*n_hidden]
            [batch_s, _, hidden_nums] = inputdata.get_shape().as_list()
            # [batch x width, 2*n_hidden]
            rnn_reshaped = tf.reshape(stack_lstm_layer, [-1, hidden_nums])

            w = tf.Variable(tf.truncated_normal([hidden_nums,
                                                self.__num_classes],
                                                stddev=0.1), name="w")
            # Doing the affine projection

            logits = tf.matmul(rnn_reshaped, w)

            logits = tf.reshape(logits, [batch_s, -1, self.__num_classes])

            raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2,
                                 name='raw_prediction')

            # Swap batch and batch axis
            # [width, batch, n_classes]
            rnn_out = tf.transpose(logits, (1, 0, 2),
                                   name='transpose_time_major')

        return rnn_out, raw_pred

    def build_shadownet(self, inputdata):
        """Build the RCNN.

        Parameters
        ----------
        inputdata : numpy ndarray
            An array containing the input data.

        Returns
        -------
        tensor (numpy ndarray)
            The built CRNN.

        """
        # first apply the cnn feature extraction stage
        cnn_out = self.__feature_sequence_extraction(inputdata=inputdata)

        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(inputdata=cnn_out)

        # third apply the sequence label stage
        net_out, raw_pred = self.__sequence_label(inputdata=sequence)
        return net_out
