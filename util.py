import glob
import os
import string

import numpy as np

sticky_def = {'NON': 'NONSTICK', '12': '12-STICKY', '34': '34-STICKY',
              '56': '56-STICKY', '78': '78-STICKY',
              'PAL': 'STICK_PALINDROME'}


def sticks_with(u, v):
    """returns 1 if two strings stick. 0 otherwise."""
    if len(u) != len(v):  # length should be same for strings to stick.
        return 0
    else:
        U = list(u)
        V = list(v)
        for i in range(len(v)):
            if U[i] == 'A':
                if V[i] != 'C':
                    return 0
            if U[i] == 'B':
                if V[i] != 'D':
                    return 0
            if U[i] == 'C':
                if V[i] != 'A':
                    return 0
            if U[i] == 'D':
                if V[i] != 'B':
                    return 0
        return 1


def sticky_type(input):
    for k in range(len(input) / 2):
        if sticks_with(input[k], input[-k - 1]) == 0:
            return sticky_index(k)


def sticky_index(k):
    """ To find out stickiness """
    if k == 0: return sticky_def['NON']
    if k in [1, 2]: return sticky_def['12']
    if k in [3, 4]: return sticky_def['34']
    if k in [5, 6]: return sticky_def['56']
    if k in [7, 8]:
        return sticky_def['78']
    else:
        return sticky_def['PAL']


def mutation(letter):
    """ returns a mutation for a letter randomly among the other letters """
    letters = "ABCD"
    random_index = np.random.randint(3)
    return letters.replace(letter, "")[random_index]


def check_line_length(text_file, n):
    """Check length of individual line in file if a length is != n,
        skip the file
    """
    with open(text_file) as f:
        for line in f:
            if len(line.strip()) != n:
                return False
        return True


def read_data(data_folder):
    files = glob.glob(data_folder)
    snippets = []
    labels = []
    for f in files:
        if os.path.isfile(f) and check_line_length(f, 40):
            # check the text and label it
            with open(f) as text_file:
                for line in text_file:
                    snippet = line.rstrip('\n')
                    snippets.append(snippet)
                    t = determine_sticky(snippet)
                    # t = sticky_type(snippet)
                    labels.append(t)
            print 'All the data in ', f, ' file are loaded'
        else:
            print 'The snippet length is not correct in ', f
    return snippets, labels


def string_to_ascii(text):
    if text is None or len(text) == 0:
        return None
    features = []
    for s in text:
        features.append(ord(s))
    return features


def numberize_output_label(text):
    if text == sticky_def['NON']:
        return [1, 0, 0, 0, 0, 0]
    elif text == sticky_def['12']:
        return [0, 1, 0, 0, 0, 0]
    elif text == sticky_def['34']:
        return [0, 0, 1, 0, 0, 0]
    elif text == sticky_def['56']:
        return [0, 0, 0, 1, 0, 0]
    elif text == sticky_def['78']:
        return [0, 0, 0, 0, 1, 0]
    elif text == sticky_def['PAL']:
        return [0, 0, 0, 0, 0, 1]


# loads data from the file
def load_test_data(data_folder):
    path = os.path.curdir + '/' + data_folder + '/*.txt'
    snippets, labels = read_data(path)
    test_x = []
    test_y = []
    for t in range(len(snippets)):
        text = snippets[t]
        label = labels[t]
        features = string_to_ascii(text)
        test_x.append(features)
        output = numberize_output_label(label)
        test_y.append(output)

    return test_x, test_y


def determine_sticky(input_str):
    """Determine the sticky label of a string"""
    if input_str is None or len(input_str) == 0:
        return

    # v, w = input_str[:len(input_str)/2], input_str[len(input_str)/2:]
    # w_reverse = w[::-1]  # reverse w
    k = 1
    is_stick = True
    while is_stick and k <= 9:
        u = input_str[:k]
        v_len = len(input_str) - 2 * k
        w = input_str[v_len + k:]
        w_reverse = w[::-1]
        is_stick = sticks_with(u, w_reverse)
        if is_stick:
            k += 1

    if k == 1:
        return sticky_def['NON']
    elif k < 9:
        if k % 2 == 0:
            return str(k - 1) + str(k) + '-STICKY'
        else:
            return str(k) + str(k + 1) + '-STICKY'
    else:
        return 'STICK_PALINDROME'


def get_correct_match(u):
    """Returns match for the char"""
    if u == 'A':
        return 'C'
    if u == 'B':
        return 'D'
    if u == 'C':
        return 'A'
    if u == 'D':
        return 'B'


def get_incorrect_match(u):
    """Returns incorrect match for the char"""
    random_num = np.random.randint(3)
    if u == 'A':
        if random_num == 0:
            return 'A'
        if random_num == 1:
            return 'B'
        if random_num == 2:
            return 'D'
    if u == 'B':
        if random_num == 0:
            return 'A'
        if random_num == 1:
            return 'B'
        if random_num == 2:
            return 'C'
    if u == 'C':
        if random_num == 0:
            return 'B'
        if random_num == 1:
            return 'C'
        if random_num == 2:
            return 'D'
    if u == 'D':
        if random_num == 0:
            return 'A'
        if random_num == 1:
            return 'C'
        if random_num == 2:
            return 'D'
