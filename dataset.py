import numpy as np
import scipy.sparse as ssp
import scipy.io
import sys,os
import os.path
import opt

class Dataset(object):
  def __init__(self, data_dir, result_dir, fn_group_info):
    self.data_dir = data_dir
    self.result_dir = result_dir
    self.fn_group_info = fn_group_info
    # load variable groups, costs, feat_dim, nbr_groups
    self.load_group_info()
    
    print 'NOTE: known data file extesions:\n'\
        'csv: comma separated file ; first column is label\n'\
        'txt: space separated file ; first column is label\n'\
        'svmlight: svm light format ; 1-based integer idx ; non integer idx are ignored \n'\
        'mat : matlab saved file with option -7, contains X: NxD, Y: Nx1 \n'\
        'npz : python saved file that contains X: NxD, Y: (N,) \n\n'
    print 'NOTE: known group info file extension: \n'\
        'csv, txt: comma/space separated file; first row is group assignment of\n'\
        '          each feature, in {0,1,2,..., nbr_groups-1}; second row is  \n'\
        '          cost of groups [ cost_grp_0, cost_grp_1, ..., cost_grp_<nbr_grp-1>]\n'\
        'npz : containing vectors: groups, costs\n'\
        'mat : containing vectors: groups, costs\n'

  def pretrain(self, fn_trains):
    X_all, Y_all = self.load_all_data(fn_trains)
    
    m_X = np.mean(X_all, axis=0)
    m_Y = np.mean(Y_all, axis=0)
    std_X = np.std(X_all, axis=0)
    std_X += std_X == 0
    X = (X_all - m_X) / std_X
    Y = (Y_all - m_Y)
    b = X.T.dot(Y) / X.shape[0]
    C_no_regul = X.T.dot(X) / X.shape[0]
    C_no_regul = (C_no_regul+C_no_regul.T) / 2
    C_no_regul += np.eye(C_no_regul.shape[0]) * 1e-10

    np.savez(self.filename_preprocess_info(fn_trains), m_X=m_X, m_Y=m_Y,
             std_X=std_X, b=b, C_no_regul=C_no_regul)

    return X, Y, b, C_no_regul, m_X, m_Y, std_X

  def pretrain_online(self, fn_trains):
    """ Only one pre_training file is opened at a time; and we don't need to
        load a full data matrix X in the memory """
    # this requrire a very different structure of how data are loaded;
    # the pretrain info are computed while the data is loaded.
    pass

  def preprocess_data(self, X_raw, Y_raw, fn_trains):
    d = np.load(self.filename_preprocess_info(fn_trains))
    m_X = d['m_X']
    m_Y = d['m_Y']
    std_X = d['std_X']
    d.close()
    X = (X_raw - m_X) / std_X
    Y = (Y_raw - m_Y)
    return X, Y
    
  def train(self, fn_trains, params):
    do_logistic = False
    d = np.load(self.filename_preprocess_info(fn_trains))
    b = d['b']
    C_no_regul = d['C_no_regul']
    X = None
    Y = None
    if ('logistic' in params) and (params['logistic']):
      do_logistic = True
      X,Y = self.load_all_data(fn_trains)
      X,Y = self.preprocess_data(X,Y, fn_trains)
      Y = Y > 0
    d.close()
    group_results = opt.all_results_bC(b, C_no_regul, params['l2_lam'], X=X, Y=Y, costs=self.costs, groups=self.groups,optimal=False, release=True, do_FR=params['do_FR'], do_logistic=do_logistic)
    np.savez(self.filename_model(fn_trains, params), **group_results)

  def predict_one_file(self, fn_test, selected, w, params):
    X_raw, Y_raw = self.load_data(fn_test)
    X, Y = self.preprocess_data(X_raw, Y_raw, fn_trains)
    
    d_preprocess_info = np.load(self.filename_preprocess_info(fn_trains))
    m_Y = d_preprocess_info['m_Y']

    if ('logistic' in params) and (params['logistic']):
      Y_hat = opt.logistic_predict(X[:,selected], w)
    else:
      Y_hat = X[:,selected].dot(w) + m_Y
    return Y_hat
  
  def compute_budget_vs_loss_one_file(self, fn_test, fn_trains, params):
    X_raw, Y_raw = self.load_data(fn_test)
    return self.compute_budget_vs_loss_XY(X_raw, Y_raw, fn_test, fn_trains, params)
 
  def compute_budget_vs_loss_multi_files(self, fn_tests, fn_trains, params):
    X_raw, Y_raw = self.load_all_data(fn_tests)
    return self.compute_budget_vs_loss_XY(X_raw, Y_raw, fn_tests, fn_trains, params)

  def compute_budget_vs_loss_XY(self, X_raw, Y_raw, fn_result, fn_trains, params):
    X, Y = self.preprocess_data(X_raw, Y_raw, fn_trains)
    do_logistic = ('logistic' in params) and (params['logistic'])
    if do_logistic:
      Y = Y > 0
    
    model = np.load(self.filename_model(fn_trains, params))
    methods = model.keys()
    budget_vs_loss_all = []
    for method_idx, method in enumerate(methods) :
      budget_vs_loss = []
      model_method = model[method]
      vec_selected = model_method['selected']
      vec_w = model_method['w']
      vec_costs = model_method['cost']
      
      for idx, cost in enumerate(vec_costs):
        selected = vec_selected[idx]
        w = vec_w[idx]
        selected_X = X[:, selected]
        if selected_X.shape[1] > 0:
          if do_logistic:
            Y_hat = opt.logistic_predict(selected_X, w)
            budget_vs_loss.append((cost, np.sum((Y_hat > 0.5) != Y) / np.float(Y.shape[0]), Y_hat))
          else:
            budget_vs_loss.append((cost, opt.loss(w, selected_X, Y)))
        else:
          if do_logistic:
            #budget_vs_loss.append(
            budget_vs_loss.append((cost, len(np.nonzero(Y)[0]) / np.float(Y.shape[0]), np.zeros(Y.shape[0])))
          else:
            budget_vs_loss.append((cost, opt.loss(0, np.zeros(Y.shape[0]), Y)))

      if do_logistic:
        budget_vs_loss = np.asarray(budget_vs_loss, 
                                    dtype=[('cost', np.float64), ('loss', np.float64), ('Y_hat', object)])
      else:
        budget_vs_loss = np.asarray(budget_vs_loss, 
                                    dtype=[('cost', np.float64), ('loss', np.float64)])
      budget_vs_loss_all.append(budget_vs_loss)

    budget_vs_loss_all = dict(zip(methods, budget_vs_loss_all)) 
    np.savez(self.filename_budget_vs_loss(fn_result, fn_trains, params), **budget_vs_loss_all)
    return budget_vs_loss_all

  def load_group_info(self):
    filename = '%s/%s' % (self.data_dir, self.fn_group_info)
    _, fextension = os.path.splitext(filename)
    if fextension == '.mat':
      d = scipy.io.loadmat(filename)
      if d['groups'].shape[1] == 1 and d['costs'].shape[1] == 1:
        self.groups = d['groups'][:, 0]
        self.costs = d['costs'][:, 0] 
      elif d['groups'].shape[0] == 1 and d['costs'].shape[0] == 1:
        self.groups = d['groups'][0, :]
        self.costs = d['costs'][0, :]
      else:
        print "Error: \"groups\" and \"costs\" must be both row vectors or both column vectors"
        sys.exit(1)

    elif fextension == '.npz':
      d = np.load(filename)
      self.groups = d['groups']
      self.costs = d['costs']
      d.close()

    elif fextension == '.csv' or fextension == '.txt':
      try:
        fin = open(filename, 'r')
      except IOError:
        print 'Error: cannot open %s' % ( filename )
        sys.exit(1)
      dlim = ',' 
      if fextension == '.txt':
        dlim = ' '
      line_idx = 0
      for l in fin :
        value_strs = l.rstrip().split(dlim)
        if line_idx == 0:
          self.groups = np.array(map(int, value_strs))
        elif line_idx == 1:
          self.costs = np.array(map(np.float64, value_strs))
        line_idx += 1
        if line_idx > 1:
          break
      fin.close()

    else:
      print "Error: Unknown file extension for group info\n"
      sys.exit(1)

    self.feat_dim = len(self.groups)
    self.nbr_groups = len(self.costs)

    # Verify that all groups are not empty. Verify each group has a cost
    for g in range(self.nbr_groups):
      if (np.max(self.groups == g) == 0):
        print "Error: There are empty groups"
        sys.exit(1)
    if np.max(self.groups) + 1 != self.nbr_groups:
      print "Error: Some groups are not assigned a cost\n"
      sys.exit(1)

  def load_all_data(self, fn_names):
    X_all = []
    Y_all = []
    filename = '%s/%s' % (self.data_dir, fn_names)
    try:
      fin = open(filename)
    except IOError:
      print 'Error: cannot open %s' % (filename)
      sys.exit(1)
    for fn in fin:  
      fn = fn.rstrip()
      X, Y = self.load_data(fn)
      X_all.append(X)
      Y_all.append(Y)
    fin.close()
    X_all = np.vstack(X_all)
    Y_all = np.hstack(Y_all)
    return X_all, Y_all

  def load_data(self, fn):
    _, fextension = os.path.splitext(fn)
    filename = '%s/%s' % (self.data_dir, fn)
    if fextension == '.mat':
      """ X is N x D and Y is N x 1 """
      d = scipy.io.loadmat(filename)
      X = d['X']
      Y = d['Y']
      if (Y.shape[0] == 1):
        Y = Y[0, :]
      elif (Y.shape[1] == 1):
        Y = Y[:, 0]
      else:
        print "Error: Y must be a vector\n"
        sys.exit(1)

    elif fextension == '.npz':
      """ X is N x D and Y is N x 1 """
      d = np.load(filename)
      X = d['X']
      Y = d['Y']
      d.close()
      if (len(Y.shape) != 1):
        print "Error: Y must be a vector\n"
        sys.exit(1)
      if (Y.shape[0] != X.shape[0]):
        print "Error: Every feature must has a label\n"
        sys.exit(1)

    elif fextension == '.csv' or fextension == '.txt':
      X = []
      Y = []
      try:
        fin = open(filename, 'r')
      except IOError:
        print 'Error: cannot open %s' % (filename)
        sys.exit(1)
      dlim = ',' 
      if fextension == 'txt':
        dlim = ' '
      for l in fin :
        features = l.rstrip().split(dlim)
        dataline = [0] * self.feat_dim
        for (i, f) in enumerate(features) :
          if i == 0 :
            Y.append(int(f))
          else:
            dataline[i-1] = np.float64(f)
        X.append(dataline)
      X = np.array(X)
      Y = np.array(Y)
      fin.close()

    elif fextension == '.svmlight':
      X = []
      Y = []
      try:
        fin = open(filename, 'r')
      except IOError:
        print 'Error: cannot open %s' % (filename)
        sys.exit(1)
      for l in fin :
        features = l.rstrip().split(' ')
        dataline = [0] * self.feat_dim
        for (i, f) in enumerate(features) :
          if i == 0 :
            Y.append(int(f))
          else:
            indfeat = f.split(':')
            try:
              dataline[int(indfeat[0])-1] = np.float64(indfeat[1])
            except ValueError:
              pass
        X.append(dataline)
      X = np.array(X)
      Y = np.array(Y)
      fin.close()

    else:
      print "Error: Unknown data file extension"
      sys.exit(1)
    return X, Y
    

  def filename_model(self, fn_trains, params):
    fn_trains = os.path.splitext(fn_trains)[0]
    return '%s/%s.model%s.npz' % (self.result_dir, fn_trains, self.param2str(params=params))

  def filename_preprocess_info(self, fn_trains):
    fn_trains = os.path.splitext(fn_trains)[0]
    return '%s/%s.pretrain.npz' % (self.data_dir, fn_trains)

  def filename_budget_vs_loss(self, fn_test, fn_trains, params):
    fn_trains = os.path.splitext(fn_trains)[0]
    fn_test = os.path.splitext(fn_test)[0]
    return '%s/%s.%s.budget_vs_loss%s.npz' % (self.result_dir, fn_test, fn_trains,
      self.param2str(params=params))

  def param2str(self,params):
    return '.lam%f' % (params['l2_lam'])

def convert_to_spams_format(X, Y, groups):
  # X, Y are preprocessed
  group_names = sorted(list(set(groups)))
  selected_feats = [np.array([], dtype=np.int)]
  selected_feats += [ np.nonzero(groups == g)[0] for g in group_names ]
  selected_feats = np.hstack(selected_feats)
  X = np.asfortranarray(X[:, selected_feats])
  Y = np.asfortranarray(Y[:, np.newaxis])
  return X, Y

def create_spams_params(groups, costs):
  spams_params = {'numThreads' : -1,'verbose' : True,
         'lambda1' : 0.001, 'it0' : 10, 'max_it' : 500,
         'L0' : 0.1, 'tol' : 1e-5, 'intercept' : False,
         'pos' : False}
  group_names = sorted(list(set(groups)))
  nbr_groups = len(group_names)
  group_sizes = [ len(np.nonzero(groups == g)[0]) for g in group_names ]
  eta_g = [ costs[g] for g in group_names ]
  eta_g = np.array([1e-9] + eta_g)
  group_sizes = [0] + group_sizes
  group_own = np.cumsum([0] + list(group_sizes)[:-1])
  group_own = group_own.astype(np.int32)
  group_sizes = np.array(group_sizes, dtype=np.int32)
  
  spams_groups = np.zeros((nbr_groups + 1, nbr_groups+1), dtype=np.bool)
  spams_groups[1:, 0] = 1
  spams_groups = ssp.csc_matrix(spams_groups, dtype=np.bool)

  spams_tree = {'eta_g' : eta_g , 'groups' : spams_groups, 
    'own_variables' : group_own, 'N_own_variables' : group_sizes }
  spams_params['compute_gram'] = True
  spams_params['loss'] = 'square'
  spams_params['regul'] = 'tree-l2'
  
  return spams_tree, spams_params

def compute_stopping_cost(alpha, d):
  d = d['OMP']
  score = d['score']
  cost = d['cost']
  alpha_score = score[np.sum( cost < 1e8 ) - 1] * alpha
  return cost[np.sum( score <= alpha_score ) - 1]

def compute_auc(costs, losses, stopping_cost):
  auc = 0
  for i in range(len(costs) - 1):
    if costs[i] >= stopping_cost:
      break
    if costs[i + 1] > stopping_cost:
      a = (stopping_cost - costs[i]) / 2.0 / (costs[i+1] - costs[i])
      auc += (losses[i+1] * a   + losses[i] * (1-a) ) * (stopping_cost - costs[i])
    else:
      auc += (costs[i + 1] - costs[i]) * (losses[i+1] + losses[i]) / 2.0
  auc /= stopping_cost * losses[0]
  return 1 - auc

def compute_oracle(costs, losses):
  c = []
  l = []
  for i in range(len(costs) - 1):
    c.append(costs[i + 1] - costs[i])
    l.append(losses[i] - losses[i + 1])
  l = np.array(l)
  c = np.array(c)
  l_over_c = 0.0 - l / c
  sorted_idx = sorted(range(len(l)), key=lambda x : l_over_c[x])
  oracle_costs = [costs[0]]
  oracle_losses = [losses[0]]
  for _, i in enumerate(sorted_idx):
    oracle_costs.append(oracle_costs[-1] + c[i])
    oracle_losses.append(oracle_losses[-1] - l[i])
  return np.array(oracle_costs), np.array(oracle_losses)
