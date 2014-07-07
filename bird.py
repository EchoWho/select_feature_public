import numpy as np
import sys
import dataset2
import opt2

def doit():
  D = dataset2.Dataset('bird_data', 'bird_results', 'groupInfo1.npz')
  fn_trains = 'toy.train.txt' #sys.argv[1] 
  fn_test = 'toy.train.txt' #sys.argv[2] 
  params={}
  params['classification'] = False
  params['l2_lam'] = 1e-6
  params['regression_methods'] = ['linear', 'glm']
  params['opt_methods'] = ['OMP']
  params['val_map'] = lambda x: x #np.log(x)
  params['val_map_predict'] = lambda x : x #np.exp(x)

  # GLM specific parameters 
  params['glm_power'] = 5
  params['glm_max_iter'] = 10

  D.train(fn_trains, params)

  D.evaluate_multi_files(fn_test, fn_trains, params)

  # params for testing with one method (regression method + optimization method)
  params['r_method'] = 'glm'
  params['o_method'] = 'OMP'
  return D.predict_one_file('out1.npz', fn_trains, params, nbr_groups=-1)
    

if __name__ == "__main__":
  doit()
