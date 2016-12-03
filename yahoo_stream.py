import stream_opt
import numpy as np
import opt_util
import opt2
import numpy as np
from generic_loader import SVMLoader
from costs_dag import CostsManager 

# load input file names
data_dir = 'yahoo_data'
fin = open('yahoo_data/fold1.train.txt')
vec_data_fn = [ l.strip() for l in fin ]
fin.close()

n_responses = 1
l2_lam = 1e-6

group_info=np.load('yahoo_data/yahoo.groups.size.10.npz')
costs = group_info['costs'][0]
groups = group_info['groups'][0]

# Feature structures. (no group each feature is in its own group) 
costs = CostsManager(costs, dep_list={}, feat_map=lambda x:x)

# dataset and preprocess for feature stats
loader = SVMLoader(700)
spd = stream_opt.StreamProblemData(n_responses, loader, data_dir, 
        vec_data_fn, costs, groups, l2_lam=l2_lam, 
        y_val_func = lambda x:x, 
        call_init=True, compute_XTY=True, 
        load_stats=True, load_dir=data_dir) 

use_linear = True
if use_linear:
    # linear regression
    solver = stream_opt.StreamOptSolverLinear(l2_lam=l2_lam, intercept=True)
else:
    # logistic regression
    solver = stream_opt.StreamOptSolverGLMExplicit(l2_lam, intercept=True, 
        mean_func=opt2.logistic_mean_func, 
        potential_func=opt2.logistic_potential, 
        gradient_func=opt2.logistic_gradient)

problem = stream_opt.StreamOptProblem(spd, solver)

result = stream_opt.alg_omp(problem, save_steps=True, step_fn_prefix='yahoo_results/step_results/step_')
np.savez('yahoo_results/omp_results.npz', result=result)

idx=-1
Y_hat, Y = solver.predict(spd, 'set1.valid.svmlight', result['model'][idx], result['selected'][idx])

print '||Y-Y_hat||^2 = {:.5e}'.format( np.sum( (Y- Y_hat)**2 ) / Y.shape[0] )
