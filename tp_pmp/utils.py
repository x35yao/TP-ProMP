
import numpy as np
import json

def force_sym(A):
    """ Returns a symmetric matrix from the received matrix
    """
    return (A + A.T) / 2.0

def make_block_diag(A, num_blocks):
    n,m = A.shape
    assert(n == m and (n % num_blocks)==0)
    block_len = n // num_blocks
    B = np.zeros((n,m))
    for i in range(n):
        for j in range(n):
            if (i // block_len) == (j // block_len):
                B[i,j] = A[i,j]
    return B

def make_close_diag(A, window_length):
    n,m = A.shape
    if window_length == 0:
        assert n == m
    else:
        assert (n == m and (n % window_length) == 0)
    B = np.zeros((n,m))
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= window_length:
                B[i,j] = A[i,j]
    return B

def numpy_serialize(obj):
    if isinstance(obj, list):
        return list(map(numpy_serialize, obj))
    elif isinstance(obj, dict):
        return {k: numpy_serialize(v) for k,v in list(obj.items())}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def lod2dol(lod):
    """ Converts a list of dictionaries to a dictionary of lists
    """
    dol = {}
    for elem in lod:
        for k,v in list(elem.items()):
            if k not in dol:
                dol[k] = []
            dol[k].append(v)
    return dol
