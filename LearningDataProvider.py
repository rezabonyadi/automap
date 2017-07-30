import numpy as np
from collections import defaultdict


def split_data(inputs, outputs, return_indexes=True, train_percent=.70, is_category=False):

    x_test_indices, x_train_indices, y_test_indices, y_train_indices = \
        extract_indexes(inputs, outputs, train_percent, is_category)

    if return_indexes is True:
        return (x_train_indices, y_train_indices), (x_test_indices, y_test_indices)
    else:
        x_train_instances, y_train_instances, x_test_instances, y_test_instances = \
            fill_instances(inputs, outputs, x_test_indices, x_train_indices, y_test_indices, y_train_indices)
        return (x_train_instances, y_train_instances), (x_test_instances, y_test_instances)


def extract_indexes(inputs, outputs, train_percent, is_category):
    n_inputs = len(inputs)

    x_train_indices = []

    if is_category is False:
        x_train_indices = np.random.choice(n_inputs, int(n_inputs * train_percent), replace=False)
    else:
        unique_cats, unique_cats_count, cat_inds = np.unique(outputs, return_counts=True, return_inverse=True)

        selected_inds = np.empty((len(unique_cats),), dtype=int)
        # Count them
        i = 0
        for cat_ind, count in cat_inds, unique_cats_count:
            selected_ind = np.random.choice(count, int(count * train_percent), replace=False)
            selected_inds[cat_ind].append(selected_ind)

        x_train_indices.append(selected_indices)
        all_inds = np.zeros(len(unique_categories))

        for i in range(n_inputs):
            cat = cat_inds[i]
            cat_ind_instance = all_inds[cat]
            selected_inds[cat][cat_ind_instance]

    y_train_indices = x_train_indices
    x_test_indices = np.array(list(set(np.arange(0, n_inputs)) - set(x_train_indices)))
    y_test_indices = x_test_indices

    return x_train_indices, y_train_indices, x_test_indices, y_test_indices


def fill_instances(inputs, outputs, x_train_indices, y_train_indices, x_test_indices, y_test_indices):
    x_train_ins = [inputs[i] for i in x_train_indices]
    y_train_ins = [outputs[i] for i in y_train_indices]
    x_test_ins = [inputs[i] for i in x_test_indices]
    y_test_ins = [outputs[i] for i in y_test_indices]

    return x_train_ins, y_train_ins, x_test_ins, y_test_ins

