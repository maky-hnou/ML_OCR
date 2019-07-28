"""Establish the char dictionary."""
import json
import os


class CharDictBuilder(object):
    """Build and read char dict."""

    def __init__(self):
        """__init__ constructor.

        Returns
        -------
        None

        """
        pass

    @staticmethod
    def write_char_dict(origin_char_list, save_path: str):
        """Write character dictionary from another char list.

        Parameters
        ----------
        origin_char_list : str
            Where the original char_list is saved.
        save_path : str
            Where to save the new char dictionary.

        Returns
        -------
        None

        """
        assert os.path.exists(origin_char_list)

        if (not save_path.endswith('.json')):
            raise ValueError('save path {:s} should be a json '
                             'file'.format(save_path))

        if (not os.path.exists(os.path.split(save_path)[0])):
            os.makedirs(os.path.split(save_path)[0])

        char_dict = dict()

        with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
            for info in origin_f.readlines():
                char_value = info[0]
                char_key = str(ord(char_value))
                char_dict[char_key] = char_value

        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(char_dict, json_f)

        origin_f.close()
        json_f.close()

    @staticmethod
    def read_char_dict(dict_path):
        """Read character dictionary.

        Parameters
        ----------
        dict_path : str
            Where dict_path is saved.

        Returns
        -------
        dict
            A dict with ord(char) as key and char as value.

        """
        assert os.path.exists(dict_path)

        with open(dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)
        json_f.close()
        return res

    @staticmethod
    def map_ord_to_index(origin_char_list, save_path):
        """Map ord of character in origin char list into index.

        Parameters
        ----------
        origin_char_list : str
            Where origin_char_list is saved.
        save_path : str
            Where the new index list will be saved.

        Returns
        -------
        None

        """
        assert os.path.exists(origin_char_list)

        if (not save_path.endswith('.json')):
            raise ValueError('save path {:s} should be a json '
                             'file'.format(save_path))

        if (not os.path.exists(os.path.split(save_path)[0])):
            os.makedirs(os.path.split(save_path)[0])

        char_dict = dict()

        with open(origin_char_list, 'r', encoding='utf-8') as origin_f:
            for index, info in enumerate(origin_f.readlines()):
                char_value = str(ord(info[0]))
                char_key = index
                char_dict[char_key] = char_value

        with open(save_path, 'w', encoding='utf-8') as json_f:
            json.dump(char_dict, json_f)

        return

    @staticmethod
    def read_ord_map_dict(ord_map_dict_path):
        """Read the ord of the character in ord_map_dict_path.

        Parameters
        ----------
        ord_map_dict_path : str
            Where ord_map_dic is saved.

        Returns
        -------
        list
            A list of chars ord.

        """
        assert os.path.exists(ord_map_dict_path)

        with open(ord_map_dict_path, 'r', encoding='utf-8') as json_f:
            res = json.load(json_f)

        return res
