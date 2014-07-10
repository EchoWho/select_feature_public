import numpy as np
import sys
import dataset 

#if __name__ == "__main__":

def doit():

  D = dataset.Dataset('grain_data', 'grain_results', 'grain.groups.npz')
  fn_trains = sys.argv[1]#'fold11.train.txt'
  params = {}
  params['l2_lam'] = 1e-6
  params['do_FR'] = False
  params['logistic'] = False

  X_tra, Y_tra, b, C_no_regul, m_X, m_Y, std_X = D.pretrain(fn_trains)
  #D.train(fn_trains, params)
  fn_test = sys.argv[2] #'ca11_no_label_bov_XY.mat'
  return D.compute_budget_vs_loss_multi_files(fn_test, fn_trains, params)
    

if __name__ == "__main__":
  doit()
