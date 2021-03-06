import numpy as np
import sys
import dataset2
import opt2

def doit():

  D = dataset2.Dataset('grain_data', 'grain_results', 'grain.groups.npz')
  #fn_trains = 'fold2.train.txt' #sys.argv[1] #e.g. 
  #fn_test = 'fold2.valid.txt' #sys.argv[2] #e.g. 
  fn_trains = 'fold%s.train.txt' % (sys.argv[1]) #e.g. 
  fn_test = 'fold%s.test.txt' % (sys.argv[1]) #e.g. 
  params={}
  params['l2_lam'] = 1e-6
  params['regression_methods'] = ['linear']
  params['opt_methods'] = ['OMP']
  params['glm_power'] = 5
  params['glm_max_iter'] = 4
  params['classification'] = True
  params['val_map'] = lambda x: x - 1
  params['val_map_predict'] = lambda x : ( x > 0.5 )

  D.train(fn_trains, params)
  return D.evaluate_multi_files(fn_test, fn_trains, params)
    

if __name__ == "__main__":
  doit()
