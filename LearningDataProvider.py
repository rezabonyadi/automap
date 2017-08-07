import random

import numpy as np


def split_data(inputs, outputs, return_indexes=True, train_percent=.70, is_category=False):

    x_test_indices, x_train_indices, y_test_indices, y_train_indices = \
        __extract_indexes__(inputs, outputs, train_percent, is_category)

    if return_indexes is True:
        return (x_train_indices, y_train_indices), (x_test_indices, y_test_indices)
    else:
        x_train_instances, y_train_instances, x_test_instances, y_test_instances = \
            __fill_instances__(inputs, outputs, x_test_indices, x_train_indices, y_test_indices, y_train_indices)
        return (x_train_instances, y_train_instances), (x_test_instances, y_test_instances)


def __extract_indexes__(inputs, outputs, train_percent, is_category):
    n_inputs = len(inputs)

    x_train_indices = []

    if is_category is False:
        x_train_indices = np.random.choice(n_inputs, int(n_inputs * train_percent), replace=False)
    else:
        unique_cats, cat_inds, unique_cats_counts = np.unique(outputs, return_counts=True, return_inverse=True)

        cats_to_indices = [[] for i in range(len(unique_cats))] #np.empty((len(unique_cats),), dtype=list)
        for i in range(n_inputs):
            cat = cat_inds[i]
            cats_to_indices[cat].append(i)

        for cats_in_index in cats_to_indices:
            unique_cats_count = len(cats_in_index)
            selected_ind = np.random.choice(unique_cats_count, int(unique_cats_count * train_percent), replace=False)
            real_ind = [cats_in_index[selected_ind[j]] for j in range(len(selected_ind))]
            x_train_indices.extend(real_ind)

    random.shuffle(x_train_indices)

    y_train_indices = x_train_indices
    x_test_indices = np.array(list(set(np.arange(0, n_inputs)) - set(x_train_indices)))
    y_test_indices = x_test_indices

    return x_train_indices, y_train_indices, x_test_indices, y_test_indices


def __fill_instances__(inputs, outputs, x_train_indices, y_train_indices, x_test_indices, y_test_indices):
    x_train_ins = [inputs[i] for i in x_train_indices]
    y_train_ins = [outputs[i] for i in y_train_indices]
    x_test_ins = [inputs[i] for i in x_test_indices]
    y_test_ins = [outputs[i] for i in y_test_indices]

    return x_train_ins, y_train_ins, x_test_ins, y_test_ins

