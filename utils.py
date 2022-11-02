# data: https://download.pytorch.org/tutorial/data.zip
import os
import unicodedata
import string
import glob

import torch
from sklearn.model_selection import train_test_split

data_dir = "data/names/"
# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.
To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.
That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def name_to_tensor(name):
    tensor = torch.zeros(len(name), 1, N_LETTERS)
    for i, letter in enumerate(name):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def load_data():
    tensor_names = []
    target_category = []
    category = [file_name.split(".")[0] for file_name in os.listdir(data_dir)]

    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file), encoding="utf8") as f:
            target = file.split(".")[0]
            names = [unicode_to_ascii(line.rstrip()) for line in f]
            for name in names:
                tensor_names.append(name_to_tensor(name))
                target_category.append(torch.tensor([category.index(target)], dtype=torch.long))
    return tensor_names, target_category, category


def retrieve_train_test_set():
    tensor_names, target_category, category = load_data()
    train_idx, test_idx = train_test_split(range(len(target_category)), test_size=0.1, shuffle=True)
    train_dataset = [(tensor_names[i], target_category[i]) for i in train_idx]
    test_dataset = [(tensor_names[i], target_category[i]) for i in test_idx]
    return train_dataset, test_dataset, category


if __name__ == '__main__':
    '''print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))

    print(letter_to_tensor('J'))  # [1, 57]
    print(name_to_tensor('Jones').size())  # [5, 1, 57]'''

    retrieve_train_test_set()
