"""Test shadow net script."""
import math
import os.path as ops

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from crnn_model.crnn_model import ShadowNet
from global_configuration import config
from local_utils.data_utils import TextFeatureIO


def test_shadownet(dataset_dir, weights_path, is_vis=False, is_recursive=True):
    """Test the CRNN.

    Parameters
    ----------
    dataset_dir : str
        Where the dataset is saved.
    weights_path : str
        Where the model is saved.
    is_vis : bool
        Whether to plot the image or not.
    is_recursive : bool
        Whether multiple images or just one image.

    Returns
    -------
    None

    """
    # Initialize the record decoder
    decoder = TextFeatureIO().reader
    images_t, labels_t, imagenames_t = decoder.read_features(
        ops.join(dataset_dir, 'test_feature.tfrecords'), num_epochs=None)
    if (not is_recursive):
        images_sh, labels_sh, imagenames_sh = \
            tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                   batch_size=32, capacity=1000+32*2,
                                   min_after_dequeue=2, num_threads=4)
    else:
        images_sh, labels_sh, imagenames_sh = \
            tf.train.batch(tensors=[images_t, labels_t, imagenames_t],
                           batch_size=32, capacity=1000 + 32 * 2,
                           num_threads=4)

    images_sh = tf.cast(x=images_sh, dtype=tf.float32)

    # build shadownet
    net = ShadowNet(phase='Test', hidden_nums=256, layers_nums=2,
                    seq_length=25, num_classes=38)

    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=images_sh)

    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, 25 * np.ones(32),
                                               merge_repeated=False)

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = \
        config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(ops.join(dataset_dir,
                                                  'test_feature.tfrecords')):
        test_sample_count += 1
    loops_nums = int(math.ceil(test_sample_count / 32))
    # loops_nums = 100

    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start predicting ......')
        if (not is_recursive):
            predictions, images, labels, imagenames = \
                sess.run([decoded, images_sh, labels_sh, imagenames_sh])
            imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
            imagenames = [tmp.decode('utf-8') for tmp in imagenames]
            preds_res = decoder.sparse_tensor_to_str(predictions[0])
            gt_res = decoder.sparse_tensor_to_str(labels)

            accuracy = []

            for index, gt_label in enumerate(gt_res):
                pred = preds_res[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if (tmp == pred[i]):
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)
                    except ZeroDivisionError:
                        if (len(pred) == 0):
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Mean test accuracy is {:5f}'.format(accuracy))

            for index, image in enumerate(images):
                print('Predict {:s} image with gt label: {:s} **** '
                      'predict label: {:s}'.format(imagenames[index],
                                                   gt_res[index],
                                                   preds_res[index]))
                if (is_vis):
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.show()
        else:
            accuracy = []
            for epoch in range(loops_nums):
                predictions, images, labels, imagenames = \
                    sess.run([decoded, images_sh, labels_sh, imagenames_sh])
                imagenames = \
                    np.reshape(imagenames, newshape=imagenames.shape[0])
                imagenames = [tmp.decode('utf-8') for tmp in imagenames]
                preds_res = decoder.sparse_tensor_to_str(predictions[0])
                gt_res = decoder.sparse_tensor_to_str(labels)

                for index, gt_label in enumerate(gt_res):
                    pred = preds_res[index]
                    totol_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if (tmp == pred[i]):
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / totol_count)
                        except ZeroDivisionError:
                            if (len(pred) == 0):
                                accuracy.append(1)
                            else:
                                accuracy.append(0)

                for index, image in enumerate(images):
                    print('Predict {:s} image with gt label: {:s} **** '
                          'predict label: {:s}'.format(imagenames[index],
                                                       gt_res[index],
                                                       preds_res[index]))
                    # if (is_vis):
                    #     plt.imshow(image[:, :, (2, 1, 0)])
                    #     plt.show()

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Test accuracy is {:5f}'.format(accuracy))

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()
    return
