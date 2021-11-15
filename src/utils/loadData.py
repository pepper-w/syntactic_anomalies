from pathlib import Path
import os
import numpy as np


def read_data_split(task, split):
    dir = Path('./data_path/' + task) # data_path required
    
    for text_file in (dir).iterdir():
      if str(os.path.basename(text_file)).endswith(split):
        if str(os.path.basename(text_file)).startswith('X'):
          with open(text_file) as f:
            texts= [x.strip() for x in f.readlines()]
        if str(os.path.basename(text_file)).startswith('labels'):
          with open(text_file) as f:
            if task != 'SOMO':
              labels = [1 if label.strip() is "I" else 0 for label in f.readlines()]
            else:
              labels = [1 if label.strip() is "C" else 0 for label in f.readlines()]

    return texts, labels


def load_lines(path):
    out = []
    with open(path, 'r') as f:
        for line in f:
            text = line.strip()
            out.append(text)
    return out


def load_labels(path, label2id):
    out = []
    with open(path, 'r') as f:
        for line in f:
            text = line.strip()
            out.append(label2id[text.upper()])
    return out


def sort_dataset(dataset):
    """
    Sort to reduce padding. Doesn't mutate original dataset.
    """
    tuples = sorted(zip(dataset['X'], dataset['y']),
                    key=lambda z:(len(z[0]), z[1]))
    sorted_dataset = {
        'X': [x for (x,y) in tuples],
        'y': [y for (x,y) in tuples]
    }    
    return sorted_dataset


def sort_group(group):
    return {k: sort_dataset(v) for k, v in group.items()}


def loadData(path, DATA, label2id):
    split_data = {'train':{}, 'dev':{}, 'test':{}}

    for key in split_data:  
        print('Loading data for {0}'.format(key))
        split_data[key] = {
            'X': load_lines(path + 'X.' + DATA + '.' + key),
            'y': load_labels(path + 'labels.' + DATA + '.' + key, label2id)
        }
        split_data[key] = sort_dataset(split_data[key])
    
    return split_data


def init_dataset():
    return {
        "X": [],
        "y": []
    }

def init_group():
    return {
      "train": init_dataset(),
      "dev": init_dataset(),
      "test": init_dataset()
    }


def merge_groups(groups):
    """
    Merge multiple groups into a single one
    """
    merged_group = init_group()
    for group in groups:
        for dataset_type in merged_group.keys():
            for k in merged_group[dataset_type].keys(): 
                merged_group[dataset_type][k] += group[dataset_type][k]
    return merged_group


def load_scramble_task(path, label2id, task):
    """
    Return data group for a single scramble task. Not sorted yet.
    """
    scramble_group = init_group()
    dataset_types = list(scramble_group.keys())
    
    label_path = os.path.join(path, "labels.{0}".format(task))
    s1_path = os.path.join(path, "s1.{0}".format(task))
    s2_path = os.path.join(path, "s2.{0}".format(task))
    
    with open(s1_path) as s1_file, open(s2_path) as s2_file, open(label_path) as label_file:
         for i, (s1, s2, label) in enumerate(zip(s1_file, s2_file, label_file)):
             dataset = scramble_group[dataset_types[i % 3]]
             dataset['X_A'].append(s1.strip())
             dataset['X_B'].append(s2.strip())
             dataset['y'].append(int(label.strip()))

    return scramble_group


def load_scramble_all(path, label2id, tasks):
    """
    Load all scramble data groups merged together as a single group, sorted to reduce padding.
    """
    groups = []
    for task in tasks:
        groups.append(load_scramble_task(path, label2id, task))
    big_group = merge_groups(groups)
    sorted_group = sort_group(big_group)
    return sorted_group




