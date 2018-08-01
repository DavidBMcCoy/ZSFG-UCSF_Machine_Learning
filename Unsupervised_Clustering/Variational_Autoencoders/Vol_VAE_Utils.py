import numpy as np

def convert_to_one_hot(Y, C):
    """
    make the label a 1x2 array for each image
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

