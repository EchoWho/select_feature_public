import stream_opt
import opt_util
import opt2
import numpy as np
from bradley_loader import BradleyLoader

loader = BradleyLoader()

data_dir = './bradley_data'
fin = open('./bradley_data/train_data_fn.txt')
vec_data_fn = [ l.strip() for l in fin ]
fin.close()

# uncomment to use all data fn
# vec_data_fn = vec_data_fn[:2]

n_responses = 1
l2_lam = 1e-4

# Full feature
group_sizes = np.hstack([np.ones(100,int) * 100, [3840, 4096, 1000]])
costs = np.hstack([ np.ones(100) * 1.7487390041351318 / 100, [0.12286496162414551, 1.1586189270019531, 1.15]])
groups = np.hstack([ np.ones(gs,int) *g  for g, gs in enumerate(group_sizes) ])

# toy 
#costs = np.ones(100)
#groups = np.arange(100)


spd = stream_opt.StreamProblemData(n_responses, loader, data_dir, 
        vec_data_fn, costs, groups, l2_lam=l2_lam, 
        y_val_func = lambda x:x, 
        call_init=True, compute_XTY=True, 
        load_stats=True, load_dir='./bradley_results') 


solver = stream_opt.StreamOptSolverLinear(l2_lam=l2_lam, intercept=True)
#solver = stream_opt.StreamOptSolverGLMExplicit(l2_lam, intercept=True, 
#    mean_func=opt2.logistic_mean_func, 
#    potential_func=opt2.logistic_potential, 
#    gradient_func=opt2.logistic_gradient)


problem = stream_opt.StreamOptProblem(spd, solver)

result = stream_opt.alg_omp(problem, save_steps=True, step_fn_prefix='bradley_results/step_result_woodbury')
np.savez('bradley_results/omp_results.npz', result=result)
