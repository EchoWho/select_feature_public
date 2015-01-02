from bradley_loader import BradleyLoader
import numpy as np


loader = BradleyLoader()

data_dir = './bradley_data'
fin = open('./bradley_data/train_data_fn.txt')
vec_data_fn = [ l.strip() for l in fin ]
fin.close()

n_X = 0
YTY = 0.0
for fn_i, fn in enumerate(vec_data_fn):
    print fn_i
    X, Y = loader.load_data(fn, lambda x:x, data_dir, load_for_train=True)
    YTY += np.dot(Y.T, Y) 
    n_X += Y.shape[0]

YTY /= n_X
print YTY
np.savez('bradley_results/YTY.npz', YTY=YTY)
    
