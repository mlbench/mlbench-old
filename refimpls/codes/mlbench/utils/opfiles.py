# -*- coding: utf-8 -*-
#
# Define the tool that will be used for other program.
#
import os
import shutil
import json
import pickle
from os.path import exists
from six.moves import cPickle


def read_text_withoutsplit(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read()


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()


def read_json(path):
    """read json file from path."""
    with open(path, 'r') as f:
        return json.load(f)


def write_txt(data, out_path, type="w"):
    """write the data to the txt file."""
    with open(out_path, type) as f:
        f.write(data)


def load_pickle(path):
    """load data by pickle."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(data, path):
    """dump file to dir."""
    print("write --> data to path: {}\n".format(path))
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def load_cpickle(path):
    """load data by pickle."""
    with open(path, 'rb') as handle:
        return cPickle.load(handle)


def write_cpickle(data, path):
    """dump file to dir."""
    print("write --> data to path: {}\n".format(path))
    with open(path, 'wb') as handle:
        cPickle.dump(data, handle)


def build_dir(path, force):
    """build directory."""
    if os.path.exists(path) and force:
        shutil.rmtree(path)
        os.mkdir(path)
    elif not os.path.exists(path):
        os.mkdir(path)
    return path


def build_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        pass


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        pass


def output_string(data, path_output, delimiter='\n'):
    """join the string in a list and output them to a file."""
    os.remove(path_output) if exists(path_output) else None

    for d in data:
        try:
            write_txt(d + delimiter, path_output, 'a')
        except:
            print(d)


def get_current_path(args, rank):
    paths = args.resume.split(',')
    splited_paths = map(
        lambda p: p.split('/')[-1].split('-')[: 1], paths)
    splited_paths_dict = dict([
        (path, paths[ind]) for ind, path in enumerate(splited_paths)])
    return splited_paths_dict[str(rank)]
