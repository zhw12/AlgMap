# https://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
# https://machinelearningmastery.com/sparse-matrices-for-machine-learning/
# https://sparse.pydata.org/en/latest/generated/sparse.COO.html#sparse.COO
'''
    Various utils
'''
import time

import numpy as np
import sparse


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def get_line_count(file):
    cnt = 0
    with open(file) as fin:
        for _ in fin:
            cnt += 1
    return cnt


def json_serializer(o):
    '''
        json serializer that handles numpy int
    '''
    if isinstance(o, np.int64): return int(o)
    raise TypeError


# get all pairs from a list
def enumerate_all_pairs(items):
    """Make all unique pairs (order doesn't matter)"""
    pairs = []
    nitems = len(items)
    for i, wi in enumerate(items):
        for j in range(i + 1, nitems):
            pairs.append((wi, items[j]))
    return pairs


# def i2w(s):
#     return [vocab[i] for i in s]

def flatten(list):
    r_list = []
    for sublist in list:
        r_list.extend(sublist)
    return r_list


# from scipy.sparse import lil_matrix


def list_to_sparse_dict(target_list):
    x = np.array(target_list)
    s = sparse.COO(x)
    result = {
        "dtype": s.dtype.str,
        "shape": s.shape,
        "data": s.data.tolist(),
        "coords": s.coords.tolist()
    }
    return result


def array_from_sparse_dict(sparse_dict):
    coords = np.array(sparse_dict['coords'])
    data = np.array(sparse_dict['data'])
    # dtype = sparse_dict['dtype']
    shape = tuple(sparse_dict['shape'])
    s = sparse.COO(coords=coords, data=data, shape=shape)
    array = s.todense()
    return array
