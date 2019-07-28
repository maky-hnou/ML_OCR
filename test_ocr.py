"""Test the RCNN."""

from tools.test_shadownet import test_shadownet

if (__name__ == '__main__'):
    # Test shadow net
    dataset_dir = './tfrec/'
    # Add the model name
    model_name = ''
    weights_path = 'model/shadownet/' + model_name
    # Chane is_recursive according whether you're testing a single or multiple
    # images
    is_recursive = True
    # Test the model
    test_shadownet(dataset_dir, weights_path, is_recursive)
