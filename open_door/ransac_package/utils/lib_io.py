
import os
import sys

import glob
import yaml
import simplejson


def makedirs(output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)


def read_yaml_file(file_path, is_convert_dict_to_class=True):
    with open(file_path, 'r') as stream:
        data = yaml.safe_load(stream)
    if is_convert_dict_to_class:
        data = dict2class(data)
    return data


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = simplejson.load(f)
    return data


def get_filenames(folder, is_base_name=False):
    ''' Get all filenames under the specific folder. 
    e.g.:
        full name: data/rgb/000001.png
        base name: 000001.png 
    '''
    full_names = sorted(glob.glob(folder + "/*"))
    if is_base_name:
        base_names = [name.split("/")[-1] for name in full_names]
        return base_names
    else:
        return full_names


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def dict2class(args_dict):
    args = SimpleNamespace()
    args.__dict__.update(**args_dict)
    return args
