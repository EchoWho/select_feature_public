import numpy as np
import sklearn.linear_model
import sys
import dataset2

import opt2

def doit():

  D = dataset2.Dataset('yahoo_data', 'yahoo_results', 
    'yahoo.groups.size.10.1sample.npz')
  fn_trains = 'fold2.train.txt' #sys.argv[1] #e.g. 
  fn_test = 'fold2.valid.txt' #sys.argv[2] #e.g. 
  params={}
  params['l2_lam'] = 1e-6
  params['regression_methods'] = ['linear']
  params['opt_methods'] = ['OMP']
  params['glm_power'] = 3

  D.train(fn_trains, params)
  return D.evaluate_multi_files(fn_test, fn_trains, params)
    

if __name__ == "__main__":
  doit()
