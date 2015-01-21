import stream_opt
import opt_util
import opt2
import numpy as np
from bradley_loader import BradleyLoader
from costs_dag import CostsManager 

loader = BradleyLoader()

data_dir = './bradley_data'
fin = open('./bradley_data/train_data_fn.txt')
vec_data_fn = [ l.strip() for l in fin ]
fin.close()

# comment out the following to use all data fn
#vec_data_fn = vec_data_fn[:100]

n_responses = 1
l2_lam = 1e-4

# Full feature
# 100 100-dim ICF, 3840 ACF, 4096 CNN_feat, 1000 CNN_prediction
group_sizes = np.hstack([np.ones(100,int) * 100, [3840, 4096, 1000]])

# no dep
#cost_list = np.hstack([ np.ones(100) * 1.7487390041351318 / 100, [0.12286496162414551, 1.1586189270019531, 1.15]])
#costs = CostsManager(cost_list, dep_list=None, feat_map = lambda x:x) 

# some dependency
#channel features: 0.3 ms
#ACF: 0.36ms + channel features
#ICF (100 feature group): 1ms + channel features
#CNN_fv6 (4096D): 5ms
#CNN_predictions (1000D): 1ms + CNN_fv6
cost_list = np.hstack([ np.ones(100) * 1, [0.36, 5, 1, 0.3]])
dep_list = {100:[103], 102:[101]}
for g in range(100):
    dep_list[g] = [103]

costs = CostsManager(cost_list, dep_list=dep_list, feat_map = lambda x:x) 

groups = np.hstack([ np.ones(gs,int) *g  for g, gs in enumerate(group_sizes) ])

# toy 
#costs = np.ones(100)
#groups = np.arange(100)


spd = stream_opt.StreamProblemData(n_responses, loader, data_dir, 
        vec_data_fn, costs, groups, l2_lam=l2_lam, 
        y_val_func = lambda x:x, 
        call_init=True, compute_XTY=True, 
        load_stats=False, load_dir='./bradley_results_ltarget') 


solver = stream_opt.StreamOptSolverLinear(l2_lam=l2_lam, intercept=True)
#solver = stream_opt.StreamOptSolverGLMExplicit(l2_lam, intercept=True, 
#    mean_func=opt2.logistic_mean_func, 
#    potential_func=opt2.logistic_potential, 
#    gradient_func=opt2.logistic_gradient)


problem = stream_opt.StreamOptProblem(spd, solver)

result = stream_opt.alg_omp(problem, save_steps=True, step_fn_prefix='bradley_results_ltarget/step_result_woodbury')
np.savez('bradley_results_ltarget/omp_results.npz', result=result)
