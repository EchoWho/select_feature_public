import stream_opt
from bradley_loader import BradleyLoader





loader = BradleyLoader()

data_dir = './bradley_data'
fin = open('./bradley_data/train_data_fn.txt')
vec_data_fn = [ l.strip() for l in fin ]

vec_data_fn = vec_data_fn[:10]
fin.close()

n_responses = 51
costs = [ 1.7487390041351318, 0.12286496162414551, 1.1586189270019531, 1.15]
groups = np.hstack( [np.zeros(10000,int), np.ones(3840,int)*1, np.ones(4096,int)*2, np.ones(1000,int)*3 ]) 

spd = stream_opt.StreamProblemData(loader, data_dir, vec_data_fn, costs, groups, l2_lam=1e-5, y_val_func = opt_util.label2indvec2, call_init=True, computeXTY=True, load_fn=None) 
