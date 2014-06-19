import dataset
import numpy as np
import sys,os


if __name__ == "__main__":

  yahoo_dataset = dataset.Dataset('grain_data', 'grain_results', 'grain.groups.npz')
  
  params = {}
  params['l2_lam'] = 1e-6
  params['do_FR'] = False
  fn_trains = 'toy_train.txt'
  print 'pretrain'
  yahoo_dataset.pretrain(fn_trains)

  print 'training'
  yahoo_dataset.train(fn_trains, params)

  print 'testing on toy '
  yahoo_dataset.compute_budget_vs_loss_one_file('set7.train.txt', fn_trains, params)
