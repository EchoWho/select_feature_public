import numpy as np
import sklearn.linear_model
import sys
import dataset 

def crowd_activity_logistic():

  D = dataset.Dataset('vision_data', 'vision_results', 'groups.mat')
  fn_trains = sys.argv[1] #e.g. 'fold11.train.txt'
  fn_test = sys.argv[2] #e.g. 'fold11.test.txt'
  params = {}
  params['l2_lam'] = 1e-5
  params['do_FR'] = False
  params['logistic'] = True

  X_tra, Y_tra, b, C_no_regul, m_X, m_Y, std_X = D.pretrain(fn_trains)
  D.train(fn_trains, params)
  return D.compute_budget_vs_loss_multi_files(fn_test, fn_trains, params)
    

if __name__ == "__main__":
  crowd_activity_logistic()
