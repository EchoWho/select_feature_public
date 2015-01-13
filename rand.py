import stream_opt
import numpy as np
import opt_util
import opt2
import numpy as np
from bradley_loader import BradleyLoader
from generic_loader import GenericLoader
from costs_dag import CostsManager 

loader = GenericLoader()

data_dir = '/home/echo/data/rand'
fin = open('/home/echo/data/rand/rand_fn.txt')
vec_data_fn = [ l.strip() for l in fin ]
fin.close()

# uncomment to use all data fn
# vec_data_fn = vec_data_fn[:2]

n_responses = 1
l2_lam = 1e-6

# Full feature
groups = np.arange(100, dtype=int)
cost_list = np.random.uniform(0.5,1.5,100)
costs = CostsManager(cost_list)

spd = stream_opt.StreamProblemData(n_responses, loader, data_dir, 
        vec_data_fn, costs, groups, l2_lam=l2_lam, 
        y_val_func = lambda x:x, 
        call_init=True, compute_XTY=True, 
        load_stats=False, load_dir='/home/echo/data/rand/results/') 


solver = stream_opt.StreamOptSolverLinear(l2_lam=l2_lam, intercept=True)
#solver = stream_opt.StreamOptSolverGLMExplicit(l2_lam, intercept=True, 
#    mean_func=opt2.logistic_mean_func, 
#    potential_func=opt2.logistic_potential, 
#    gradient_func=opt2.logistic_gradient)


problem = stream_opt.StreamOptProblem(spd, solver)

result = stream_opt.alg_omp(problem, save_steps=True, step_fn_prefix='/home/echo/data/rand/results/step_result')
np.savez('/home/echo/data/rand/results/omp_results.npz', result=result)
