import numpy as np
import sys
import dataset2
import opt2
import opt_util

def doit():

  D = dataset2.Dataset('mnist_data', 'mnist_results', 'group_info_724.npz')
  fn_trains = 'train724.txt' #sys.argv[1] #e.g. 
  fn_test = 'test724.txt' #sys.argv[2] #e.g. 
  params={}
  params['l2_lam'] = 1e-5
  params['regression_methods'] = ['logistic']
  params['opt_methods'] = ['OMP', 'OMP_stacked']
  params['glm_power'] = 5
  params['glm_max_iter'] = 10
  params['classification'] = True
  params['val_map'] = opt_util.label2indvec
  params['val_map_predict'] = opt_util.indvec2label

  D.train(fn_trains, params)
  return D.evaluate_multi_files(fn_test, fn_trains, params)
    

if __name__ == "__main__":
  ret = doit()
  np.savez('grain_results/grain2.py.results.9.26.py', ret=ret)
