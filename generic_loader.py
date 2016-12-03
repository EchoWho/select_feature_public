import numpy as np
import scipy
import dataset2


class GenericLoader(object):
    def __init__(self):
        pass

    def load_data(self, fn, y_val_func=lambda x:x, data_dir='.', load_for_train=False):
        fin = np.load('{}/{}'.format(data_dir, fn))
        X = fin['X']
        Y = fin['Y']
        return X, y_val_func(Y)

class SVMLoader(object):

    def __init__(self, data_dim):
        self.data_dim = data_dim
        

    def load_data(self, fn, y_val_func=lambda x:x, data_dir='.', load_for_train=False):
        return dataset2.load_data(fn, y_val_func, self.data_dim, data_dir)  
