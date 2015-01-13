import numpy as np
import numpy
import time
import sys
import functools
import scipy.linalg
import opt_util
import pdb
import scipy.linalg

def square_error(Y_hat, Y):
  return np.sum((Y_hat - Y)**2) / np.float64(Y.shape[0])

def logistic_mean_func(dot_Xw):
  if dot_Xw.shape[1] == 1:
    exp_Xw = np.exp(-dot_Xw)
    return 1.0 / (1.0 + exp_Xw)
  max_dot_Xw = np.max(dot_Xw, axis=1)[:,np.newaxis]
  exp_Xw = np.exp(dot_Xw - max_dot_Xw)
  normalizer = np.sum(exp_Xw, axis=1)[:,np.newaxis]
  return exp_Xw / normalizer

def logistic_gradient(dot_Xw):
  if dot_Xw.shape[1] == 1:
    exp_Xw = np.exp(-dot_Xw)
    return (exp_Xw / (1.0 + exp_Xw)**2)[:, np.newaxis, :]
  max_dot_Xw = np.max(dot_Xw, axis=1)[:,np.newaxis]
  exp_Xw = np.exp(dot_Xw - max_dot_Xw)
  Z = np.sum(exp_Xw, axis=1)[:, np.newaxis]
  exp_Xw /= Z
  N = dot_Xw.shape[0]
  K = dot_Xw.shape[1]
  # (N, K, K) gradient. 
  return exp_Xw[:,:,np.newaxis] * (np.tile(np.eye(K), (N, 1)).reshape(N, K, K) - exp_Xw[:,np.newaxis,:])

def logistic_potential(dot_Xw):
  if dot_Xw.shape[1] == 1:
    return np.log(1.0 + np.exp(-dot_Xw)) + dot_Xw
  max_dot_Xw = np.max(dot_Xw, axis=1)[:,np.newaxis]
  exp_Xw = np.exp(dot_Xw - max_dot_Xw)
  return np.log(np.sum(exp_Xw, axis=1)[:, np.newaxis]) + max_dot_Xw

def logistic_lipschitz():
  return 1

def opt_linear(X,Y,C_inv=None):
  b = X.T.dot(Y) / np.float64(Y.shape[0])
  if C_inv == None:
    C = X.T.dot(X) / np.float64(Y.shape[0])
    C = (C + C.T) / 2.0
    C_inv = np.linalg.pinv(C)
  return C_inv.dot(b)

# X : (N, D)
# Y : (N, K)
# potential_func : (N, K) -> (N, 1) // part of the objective
# mean_func : (N, K) -> (N, K)   // gradient of the potential and E[Y | X] = mean(X dot w) 
# w0 : (D, K)   // warm start weight vector for( X dot w)
# C_inv : // inverse of X.T.dot(X). 
#
# Typical use: logistic regression
def opt_glm_explicit(X, Y, potential_func, mean_func, w0=None,
                     intercept=True, C_inv=None, l2_lam=None):
  nbr_samples = np.float64(Y.shape[0])
  nbr_feats = X.shape[1]
  nbr_responses = Y.shape[1]
  w_len = nbr_feats + np.int32(intercept)

  if l2_lam == None:
    l2_lam = 1e-6

  if C_inv == None:
    C_no_regul = X.T.dot(X) / nbr_samples
    C_no_regul = (C_no_regul + C_no_regul.T) / 2.0
    C = C_no_regul + np.eye(C_no_regul.shape[0]) * l2_lam
    C_inv = np.linalg.pinv(C)

  if intercept:
    C_inv_tmp = np.zeros((w_len, w_len))
    C_inv_tmp[1: , 1:] = C_inv
    C_inv = C_inv_tmp
    C_inv[0,0] = 1
    X = np.hstack([np.ones((Y.shape[0], 1)), X])

  w = np.zeros((w_len, nbr_responses))
  if w0 != None:
    w[:w0.shape[0]] = w0
  
  has_converge = False
  nbr_iter = -1
  beta = 0.8
  while True:
    nbr_iter += 1
    dot_Xw = X.dot(w)
  
    objective = (np.sum(potential_func(dot_Xw)) - np.sum(Y * dot_Xw)) / nbr_samples
    if intercept:
      objective += l2_lam * np.sum( (w[1:] * w[1:]) ) / 2.0
    else:
      objective += l2_lam * np.sum( (w * w) ) / 2.0
    print "iteration: {}. objective: {}".format(nbr_iter, objective)

    if nbr_iter > 0:
      conv_num = abs(last_objective - objective) / np.abs(last_objective) 
      #gcp.info("conv num: {}".format(conv_num))
      has_converge = conv_num < 1e-5
      if has_converge:
        break
      if last_objective < objective:
        #gcp.info("iteration: {}. Step size was too large. Shrinking!!!".format(nbr_iter))
        total_delta_w *= beta
        w = last_w - total_delta_w
        continue

    last_objective = objective
    last_w = w.copy()

    pred = mean_func(dot_Xw)
    residual = (pred  - Y) / nbr_samples 
    L_lipschitz = np.max(abs(w)) * l2_lam + 1
    delta_w = (1 / L_lipschitz) * C_inv.dot(X.T.dot(residual))
    regul_delta_w = w * l2_lam
    if intercept:
      regul_delta_w[0] = 0
    total_delta_w = delta_w + regul_delta_w
    w -= total_delta_w

  return w, -objective 
      
# X : (N,D)
# Y : (N,K)
# calib_funcs : [ (N, K) -> (N, K) ]  // a list of "mean functions" that are to be linear combined 
#               to form the final prediction function 
def opt_glm_implicit(X, Y, calib_funcs, max_iter=None, intercept=True, C_inv=None, l2_lam=None):
  nbr_calib_funcs = len(calib_funcs)
  nbr_samples = np.float64(Y.shape[0])
  nbr_feats = X.shape[1]
  if len(Y.shape) == 1:
    Y = Y[:,np.newaxis]
  nbr_responses = Y.shape[1]
  w_len = nbr_feats + intercept
  if l2_lam == None:
    l2_lam = 1e-6

  if C_inv == None:
    C_no_regul = X.T.dot(X) / nbr_samples
    C_no_regul = (C_no_regul + C_no_regul.T) / 2.0
    C = C_no_regul + np.eye(C_no_regul.shape[0]) * l2_lam
    C_inv = np.linalg.inv(C)

  if intercept:
    C_inv_tmp = np.zeros((w_len, w_len))
    C_inv_tmp[1: , 1:] = C_inv
    C_inv = C_inv_tmp
    C_inv[0,0] = 1
    X = np.hstack([np.ones((nbr_samples, 1)), X])

  vec_w = []
  vec_w_tilt = []
  w = np.zeros((w_len, nbr_responses))
  Y_hat = np.zeros(Y.shape)
  vec_w.append(w)

  has_converge = False
  is_first = True
  iter_idx = 0
  while (not has_converge) and (max_iter == None or iter_idx < max_iter):
    # Linear fit residual
    w = opt_linear(X, Y-Y_hat, C_inv)
    vec_w.append(w)
    Y_tilt = Y_hat + X.dot(w)
    
    # linear fit target Y with current predictions cailb_func(Y_tilt) 
    G_Y_tilt = np.array([ f(Y_tilt).ravel() for f in calib_funcs ]).T
    w_tilt, _, _, _ = np.linalg.lstsq(G_Y_tilt, Y.ravel())
    vec_w_tilt.append(w_tilt) 
    Y_hat = G_Y_tilt.dot(w_tilt).reshape(Y.shape)

    # Use square loss to determine convergence
    loss = np.sum((Y_hat - Y)**2) / nbr_samples
    if is_first:
      loss_init = np.sum(Y**2) / nbr_samples
      is_first = False
    elif abs(old_loss - loss) < 1e-5:
      has_converge = True
      break
    old_loss = loss
    iter_idx += 1
    #print "nbr_feats %d, Iter %d, loss %f" % (X.shape[1], iter_idx, loss )
  model = (vec_w, vec_w_tilt)
  return model, loss_init - loss

def glm_explicit_predict(X, w, mean_func, intercept=True):
  if intercept:
    dot_Xw = X.dot(w[1:]) + w[0]
  else:
    dot_Xw = X.dot(w)
  return mean_func(dot_Xw)

def glm_implicit_predict(X, vec_w, vec_w_tilt, calib_funcs, intercept=True, 
                         compute_grad=False, Y=None, X_full=None, 
                         weighted_gradients=None):
  T = len(vec_w_tilt)
  nbr_responses = vec_w[0].shape[1]
  nbr_samples = X.shape[0]
  nbr_feats = X.shape[1]
  if intercept:
    X = np.hstack([np.ones((nbr_samples, 1)), X])
  if compute_grad:
    vec_Y_tilt = np.zeros((T, nbr_samples, nbr_responses))
  Y_hat = X.dot(vec_w[0])
  for t, w_tilt in enumerate(vec_w_tilt):
    Y_hat += X.dot(vec_w[t+1]) #Y_tilt
    if compute_grad:
      vec_Y_tilt[t] = Y_hat # Y_tilt recorded
    Y_hat = np.array([ f(Y_hat) for f in calib_funcs ]).T.dot(w_tilt).T

  if compute_grad:
    if Y == None:
      print "Errror : Need Ground truth for computing gradient"
      sys.exit(1)
    if len(Y.shape)==1:
      Y = Y[:,np.newaxis]
    U = Y_hat - Y
    t = T - 1

    if T == 0:
      w_grads = np.zeros((1, X_full.shape[1], nbr_responses),np.float64)
      w_grads[0] = X_full.T.dot(U)
    else:
      w_grads = np.zeros((T, X_full.shape[1], nbr_responses),np.float64) # note that w0 and w1 are merged together. 
    while t >= 0:
      U = weighted_gradients(Y_tilt=vec_Y_tilt[t], w_tilt=vec_w_tilt[t], Z=U)
      w_grads[t] = X_full.T.dot(U)
      t -= 1
    return Y_hat, w_grads 

  return Y_hat

def generate_glm_funcs(K, P):
  nbr_classes = K
  calib_funcs = []
  gradients = []
  if K == 1:
    for p in range(P):
      if p == 0:
        calib_funcs.append(lambda X : np.ones(X.shape))
      else:
        calib_funcs.append(lambda X, p=p : X**p)
  else:
    def k_th_column_matrix(col_vec, k, K):
      z = np.zeros((col_vec.shape[0], K))
      z[:,k] = col_vec
      return z
    calib_funcs.append(lambda X : X)
    for k in range(K):
      for p in range(P):
        calib_funcs.append(lambda X,k=k,p=p : k_th_column_matrix(X[:,k]**p, k, X.shape[1]))
  return calib_funcs, weighted_calib_func_gradients(K,P)

def weighted_calib_func_gradients(K, P):
  def w_tilt_grad_G_internal(Y_tilt, w_tilt, Z, K, P):
    if K == 1:
      U = np.zeros(Z.shape)
      for p in range(P-1):
        U += Y_tilt ** p * Z * w_tilt[p+1]
    else:
      U = w_tilt[0] * Z
      w_tilt_idx = 1
      for k in range(K):
        u_k = np.zeros(z.shape)
        y_k = Y_tilt[:,k]
        for p in range(P):
          if p != 0:
            u_k += y_k**(p-1) * w_tilt[w_tilt_idx]
          w_tilt_idx += 1
        U[:,k] += u_k * Z[:,k]
    return U
  return functools.partial(w_tilt_grad_G_internal, K=K, P=P)

def alg_forward(problem, K=None, costs=None, groups=None):
  n_features = problem.n_features()

  if groups is None:
    groups = np.arange(n_features)
  n_groups = np.max(groups) + 1

  if K is None:
    K = n_groups

  if costs is None:
    costs = np.ones(n_groups)

  mask = np.zeros(n_groups, np.bool)
  selected_groups = np.zeros(K, np.int)
  best_feats = np.zeros(n_features, np.int)
  best_feats_end = 0
  last_score = 0.0
  last_model = problem.init_model()
  sequence = [(0.0, 0.0, -1, [], [], last_model, 0)]

  t0 = time.time()
  for k in range(K):
    print 'FR Iteration %d' % k
    best_gain = 0
    best_group = -1
    for g in range(n_groups):
      # If feature is already selected just skip it
      if mask[g]:
        continue

      feats_g = problem.data.vec_feats_g[g]
      sel_feats_end = best_feats_end + feats_g.shape[0]
      best_feats[best_feats_end:sel_feats_end] = feats_g
      # Experiment shows that warm start doesn't improve the results. 
      #model, score = problem.opt_and_score(best_feats[:sel_feats_end], last_model)
      model, score = problem.opt_and_score(best_feats[:sel_feats_end])
      gain = (score - last_score) / costs[g]

      if (gain > best_gain):
        best_gain = gain
        best_group = g
        best_model = model
        best_score = score
    
    if best_group == -1:
      print 'Exited with no group selected on iteration %d' % (k+1)
      break

    mask[best_group] = True
    selected_groups[k] = best_group
    best_groups = selected_groups[:k+1]

    feats_g = problem.data.vec_feats_g[best_group]
    sel_feats_end = best_feats_end + feats_g.shape[0]
    best_feats[best_feats_end:sel_feats_end] = feats_g
    best_feats_end = sel_feats_end

    last_score = best_score
    last_model = best_model
    c = np.sum(costs[best_groups])

    timestamp = time.time() - t0
    sequence.append((last_score, c, best_group, best_feats[:best_feats_end], 
                     best_groups, best_model, timestamp))

    # Hack for AISTATS save the steps one by one.
    #np.savez("mnist_results/FR_{}.npz".format(k), cost=c, group=best_group, selected=best_feats[:best_feats_end],
    #         selected_groups=best_groups, model=best_model, time=timestamp)

  return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                    ('selected', object), ('selected_groups', object),
                    ('model', object), ('time', np.float)])

def alg_omp(problem, K=None, costs=None, groups=None):
  n_features = problem.n_features()
  if groups is None:
    groups = np.arange(n_features)
  n_groups = np.max(groups) + 1
  if K is None:
    K = n_groups
  if costs is None:
    costs = np.ones(n_groups)

  mask = np.zeros(n_groups, np.bool)
  selected_groups = np.zeros(K, np.int)
  best_feats = np.zeros(n_features, np.int)
  best_feats_end = 0
  last_model=problem.init_model()
  sequence = [(0.0, 0.0, -1, [], [], last_model, 0)]

  t0 = time.time()
  for k in range(K):
    print 'OMP Iteration %d' % k

    g = problem.omp_select_groups(best_feats[:best_feats_end], mask, 
      last_model, costs, groups)
    if g == -1:
      print 'Exited with no group selected on iteration %d' % (k+1)
      break

    selected_groups[k] = g
    mask[g] = True

    best_groups = selected_groups[:k+1]

    feats_g = problem.data.vec_feats_g[g]
    sel_feats_end = best_feats_end + feats_g.shape[0]
    best_feats[best_feats_end:sel_feats_end] = feats_g
    best_feats_end = sel_feats_end

    #last_model, best_score = problem.opt_and_score(best_feats[:best_feats_end], last_model) 
    last_model, best_score = problem.opt_and_score(best_feats[:best_feats_end]) 
    c = np.sum(costs[best_groups])

    #print best_score

    timestamp = time.time() - t0
    sequence.append((best_score, c, g, best_feats[:best_feats_end], 
                     best_groups, last_model, timestamp))

  return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                    ('selected', object), ('selected_groups', object),
                    ('model', object), ('time', np.float)])

def alg_stacked_omp(problem, K=None, costs=None, groups=None):
  nbr_chunks = problem.data.chunks.shape[0]
  n_features = problem.n_features()
  if groups is None:
    groups = np.arange(n_features)
  n_groups = np.max(groups) + 1
  if K is None:
    K = n_groups
  if costs is None:
    costs = np.ones(n_groups)

  mask = np.zeros(n_groups, np.bool)
  selected_groups = np.zeros(K, np.int)
  best_feats = np.zeros(n_features, np.int)
  best_feats_end = 0
  last_model = problem.init_model()
  last_models = [ problem.init_model() for i in range(nbr_chunks) ]
  sequence = [(0.0, 0.0, -1, [], [], last_model, 0)]

  t0 = time.time()
  for k in range(K):
    print 'OMP Iteration %d' % k

    g = problem.stacked_omp_select_groups(best_feats[:best_feats_end], mask, 
      last_models, costs, groups)
    if g == -1:
      print 'Exited with no group selected on iteration %d' % (k+1)
      break

    selected_groups[k] = g
    mask[g] = True

    best_groups = selected_groups[:k+1]

    feats_g = problem.data.vec_feats_g[g]
    sel_feats_end = best_feats_end + feats_g.shape[0]
    best_feats[best_feats_end:sel_feats_end] = feats_g
    best_feats_end = sel_feats_end

    # Warm start doesn't work well on GEM feature on MNIST and GRAIN. So we removed it.
    # last_model, best_score = problem.opt_and_score(best_feats[:best_feats_end], last_model) 
    last_model, best_score = problem.opt_and_score(best_feats[:best_feats_end]) 
    c = np.sum(costs[best_groups])

    #print best_score

    # Update last_models (models for estimating the gradient on test sets)
    for ci in range(nbr_chunks):
      last_models[ci], _ = problem.opt_and_score(best_feats[:best_feats_end], chunk_i=ci) 

    timestamp = time.time() - t0
    sequence.append((best_score, c, g, best_feats[:best_feats_end], 
                     best_groups, last_model, timestamp))

  return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                    ('selected', object), ('selected_groups', object),
                    ('model', object), ('time', np.float)])


class ProblemData(object):
  def __init__(self, X=None, Y=None, costs=None, groups=None, l2_lam=1e-6, nbr_chunks=5):
    if X==None and Y==None and C_no_regul==None and b==None:
      print "Error: no data are given"
    self.X = X
    if len(Y.shape) == 1:
      self.nbr_responses = 1
      Y = Y.reshape((Y.shape[0], 1))
    else:
      self.nbr_responses = Y.shape[1]
    self.Y = Y
    self.m_Y = np.mean(Y, axis=0)

    self.b = X.T.dot(Y) / np.float64(X.shape[0]) 
    C_no_regul = X.T.dot(X) / np.float64(X.shape[0])
    C_no_regul = (C_no_regul.T + C_no_regul) * 0.5
    self.C_no_regul = C_no_regul

    self.groups = groups
    if groups == None:
      self.groups = np.arange(C_no_regul.shape[0])

    self.costs = costs
    if costs == None:
      self.costs = np.ones(C_no_regul.shape[0])

    # Per group covariance
    n_groups = np.max(self.groups)+1
    self.vec_feats_g = []
    self.group_C_invs = []
    for g in range(n_groups):
      feats_g = np.nonzero(self.groups==g)[0]
      self.vec_feats_g.append(feats_g)
      C_g = self.C_no_regul[feats_g[:,np.newaxis], feats_g] 
      self.group_C_invs.append(np.linalg.pinv(C_g)) 

    # For each chunk compute b, C, C_g
    self.chunks, self.c_chunks = opt_util.generate_chunks(Y.shape[0], nbr_chunks)
    self.chunk_C_no_regul = []
    self.chunk_b = []
    self.c_chunk_C_no_regul = []
    self.c_chunk_b = []
    self.chunk_group_C_invs = []
    for ci in range(nbr_chunks):
      chunk = self.chunks[ci, :]
      c_chunk = self.c_chunks[ci, :]

      # b = XY
      b = X[chunk,:].T.dot(Y[chunk,:]) / np.float64(Y.shape[0])
      self.chunk_b.append(b)
      self.c_chunk_b.append(self.b - b)

      # C_no_regual
      C_no_regul = X[chunk,:].T.dot(X[chunk,:]) / np.float64(Y.shape[0])
      C_no_regul = (C_no_regul.T + C_no_regul) * 0.5
      self.chunk_C_no_regul.append(C_no_regul) 
      self.c_chunk_C_no_regul.append(self.C_no_regul - C_no_regul) 

      # C_g
      self.chunk_group_C_invs.append([])
      for _, feats_g in enumerate(self.vec_feats_g):
        C_g = C_no_regul[feats_g[:,np.newaxis], feats_g]
        self.chunk_group_C_invs[-1].append(np.linalg.pinv(C_g))

    self.set_l2_lam(l2_lam)

  def n_samples(self):
    return self.X.shape[0]

  def n_features(self):
    return self.C_no_regul.shape[0]

  def n_responses(self):
    return self.nbr_responses

  def set_l2_lam(self, l2_lam):
    self.l2_lam = l2_lam
    diag_C_no_regul = self.C_no_regul.diagonal().copy()
    diag_C_no_regul += diag_C_no_regul == 0 
    self.C = self.C_no_regul + np.diag(diag_C_no_regul * l2_lam, 0)

    self.c_chunk_C = []
    self.chunk_C = []
    for ci, c_chunk_C in enumerate(self.c_chunk_C_no_regul):
      diag_C_no_regul = c_chunk_C.diagonal().copy()
      diag_C_no_regul += diag_C_no_regul == 0 
      self.c_chunk_C.append(c_chunk_C + np.diag(diag_C_no_regul * l2_lam, 0))

      chunk_C = self.chunk_C_no_regul[ci]
      diag_C_no_regul = chunk_C.diagonal().copy()
      diag_C_no_regul += diag_C_no_regul == 0 
      self.chunk_C.append(chunk_C + np.diag(diag_C_no_regul * l2_lam, 0))

class OptSolverLinear(object):
  def __init__(self, l2_lam=1e-6, intercept=True):
    self.l2_lam = l2_lam
    self.intercept=intercept

  def opt_and_score(self, data, selected_feats, model0=None, chunk_i=None):
    if chunk_i == None:
      dataC = data.C
      datab = data.b
    else:
      dataC = data.c_chunk_C[chunk_i]
      datab = data.c_chunk_b[chunk_i]

    C = dataC[selected_feats[:,np.newaxis], selected_feats]
    b = datab[selected_feats]
    if self.intercept:
      C_tmp = np.zeros((C.shape[0] + 1, C.shape[1] + 1))
      C_tmp[1:,1:] = C
      C_tmp[0,0] = 1
      C = C_tmp
      b_tmp = np.zeros((b.shape[0] + 1, b.shape[1]))
      b_tmp[1:] = b
      b_tmp[0] = data.m_Y
      b = b_tmp
    w = np.linalg.pinv(C).dot(b)
    return w, np.sum((2 * b - C.dot(w)) * w - self.l2_lam * w * w)

  @staticmethod
  def predict(X, model, intercept=True):
    if intercept:
      return X.dot(model[1:]) + model[0] 
    return X.dot(model)

  def init_model(self, nbr_responses):
    return np.zeros((self.intercept, nbr_responses))

  def compute_grad_proxy(self, data, selected_feats, model, chunk_i=None):
    if chunk_i == None:
      dataC = data.C
      datab = data.b
    else:
      dataC = data.chunk_C[chunk_i]
      datab = data.chunk_b[chunk_i]

    if self.intercept:
      return datab - dataC[:,selected_feats].dot(model[1:]) - model[0, :]
    return datab - dataC[:,selected_feats].dot(model)

  def compute_whitened_group_gradient_square(self, grad_proxy, data, sel_g, M, chunk_i=None):
    b_g = grad_proxy[sel_g]
    return np.sum(b_g.T.dot(M) * b_g.T) / np.float64(data.n_features())

class OptSolverLogistic(object):
  def __init__(self, l2_lam=1e-6, intercept=True):
    self.l2_lam = l2_lam
    self.intercept = intercept

  def opt_and_score(self, data, selected_feats, model0=None, chunk_i=None):
    if chunk_i == None:
      dataC = data.C
      data_range = np.arange(data.n_samples())
    else:
      dataC = data.c_chunk_C[chunk_i]
      data_range = data.c_chunks[chunk_i, :]

    C_inv = scipy.linalg.inv, dataC[selected_feats[:,np.newaxis], selected_feats]
    return opt_glm_explicit(data.X[data_range[:, np.newaxis], selected_feats], data.Y[data_range, :], 
                            logistic_potential, logistic_mean_func,  
                            w0=model0, 
                            C_inv=C_inv,
                            intercept=self.intercept, l2_lam=self.l2_lam)

  @staticmethod
  def predict(X, model, intercept=True):
    if intercept:
      return logistic_mean_func(X.dot(model[1:]) + model[0])
    return logistic_mean_func(X.dot(model))

  def init_model(self, nbr_responses):
    return np.zeros((self.intercept, nbr_responses), dtype=np.float64)
  
  def compute_grad_proxy(self, data, selected_feats, model, chunk_i=None):
    if chunk_i == None:
      data_range = np.arange(data.n_samples())
    else:
      data_range = data.chunks[chunk_i, :]

    sel_X = data.X[data_range[:, np.newaxis],selected_feats]
    if self.intercept:
      dot_Xw = sel_X.dot(model[1:]) + model[0]
    else:
      dot_Xw = sel_X.dot(model)
    # res = (data.Y - logistic_mean_func(dot_Xw)) 
    # note that each logsitic_gradients is a symmetric KxK, so we use axis=2 or 1.
    # essenstially each residual apply the KxK linear tansf of gradient.
    # result is NxK 
    return np.sum((data.Y[data_range, :] - logistic_mean_func(dot_Xw))[:, np.newaxis, :] * logistic_gradient(dot_Xw), axis=2)

  def compute_whitened_group_gradient_square(self, grad_proxy, data, sel_g, M, chunk_i=None):
    if chunk_i == None:
      data_range = np.arange(data.n_samples())
    else:
      data_range = data.chunks[chunk_i, :]

    b_g = grad_proxy.T.dot(data.X[data_range[:,np.newaxis],sel_g]) 
    return np.sum((b_g.dot(M)) * b_g) / np.float64(data.n_features())

class OptSolverGLM(object):
  def __init__(self, l2_lam=1e-6, glm_power=4, nbr_responses=1, max_iter=None):
    self.l2_lam = l2_lam
    self.glm_power = glm_power
    self.nbr_responses = nbr_responses
    self.intercept=True
    self.max_iter=max_iter
    self.calib_funcs, self.weighted_gradients_func = \
      generate_glm_funcs(nbr_responses, glm_power)

  def opt_and_score(self, data, selected_feats, model0=None, chunk_i=None):
    if chunk_i == None:
      C = data.C
      data_range = np.arange(data.n_samples())
    else:
      C = data.c_chunk_C[chunk_i] 
      data_range = data.c_chunks[chunk_i, :]

    return opt_glm_implicit(data.X[data_range[:, np.newaxis], selected_feats], data.Y[data_range, :], self.calib_funcs, 
      max_iter=self.max_iter, 
      C_inv=np.linalg.pinv(C[selected_feats[:,np.newaxis], selected_feats]))

  @staticmethod
  def predict(X, model, calib_funcs):
    if X.shape[1] > 0:
      return glm_implicit_predict(X, model[0], model[1], 
        calib_funcs, compute_grad=False)
    else:
      return np.zeros((X.shape[0], model[0][0].shape[1]))

  def init_model(self, nbr_responses):
    return ([np.zeros((self.intercept, nbr_responses), dtype=np.float64)], [])

  def compute_grad_proxy(self, data, selected_feats, model, chunk_i=None):
    if chunk_i == None:
      data_range = np.arange(data.n_samples())
    else:
      data_range = data.chunks[chunk_i, :]

    Y_hat, w_grads = \
      glm_implicit_predict(data.X[data_range[:, np.newaxis], selected_feats], model[0], model[1], 
        self.calib_funcs, compute_grad=True, Y=data.Y[data_range, :], X_full=data.X,
        weighted_gradients=self.weighted_gradients_func)
    return w_grads

  def compute_whitened_group_gradient_square(self, w_grads, data, sel_g, M, chunk_i=None):
    w_grads_g = np.transpose(w_grads[:, sel_g, :], axes=[0,2,1])
    return np.sum(w_grads_g.dot(M) * w_grads_g)

class OptProblem(object):
  def __init__(self, problem_data, opt_solver):
    self.data = problem_data
    self.solver = opt_solver

  def set_solver(self, solver):
    self.solver = solver

  def n_features(self):
    return self.data.n_features()

  def opt_and_score(self, selected_feats, model0=None, chunk_i=None):
    return self.solver.opt_and_score(self.data, selected_feats, model0, chunk_i)

  def init_model(self):
    return self.solver.init_model(self.data.n_responses())

  def omp_select_groups(self, selected_feats, mask, model, costs, groups):
    grad_proxy = self.solver.compute_grad_proxy(self.data, selected_feats, model)

    best_ip = 0.0
    best_g = -1

    #print 'running omp selection with %s already selected' % np.sum(mask)

    for g in np.unique(groups):
      if mask[g]:
        continue

      sel_g = np.nonzero(groups == g)[0]
      whitened_grad_square = self.solver.compute_whitened_group_gradient_square(\
        grad_proxy, self.data, sel_g, self.data.group_C_invs[g])
      ip = whitened_grad_square / costs[g]

      #print 'group %d ip %f' % (g, ip)
      #print ip, np.sum(mask)

      if ip > best_ip:
        best_ip = ip
        best_g = g

    #print 'best was %d ip %f' % (best_g, best_ip)
    return best_g

  def stacked_omp_select_groups(self, selected_feats, mask, models, costs, groups):

    grad_proxy = []
    for ci, model in enumerate(models):
      grad_proxy.append(self.solver.compute_grad_proxy(self.data, selected_feats, model, chunk_i=ci))

    best_ip = 0.0
    best_g = -1

    #print 'running omp selection with %s already selected' % np.sum(mask)

    for g in np.unique(groups):
      if mask[g]:
        continue

      sel_g = np.nonzero(groups == g)[0]
      whitened_grad_square = 0
      for ci in range(len(models)):
        whitened_grad_square += self.solver.compute_whitened_group_gradient_square(\
          grad_proxy[ci], self.data, sel_g, self.data.chunk_group_C_invs[ci][g], chunk_i=ci)
      ip = whitened_grad_square / costs[g]

      #print 'group %d ip %f' % (g, ip)
      #print ip, np.sum(mask)

      if ip > best_ip:
        best_ip = ip
        best_g = g

    #print 'best was %d ip %f' % (best_g, best_ip)
    return best_g



def all_results(X=None, Y=None, 
                costs=None, groups=None,
                K=None,
                l2_lam=1e-6, 
                regression_methods=['linear'],  # 'logistic', 'glm'
                opt_methods=['FR'], # 'OMP'
                params={}):
 
  data = ProblemData(X,Y,costs,groups,l2_lam, nbr_chunks=10)
  ret = [] 
  for rm in regression_methods:
    # Initialize problem (problem data + solver)
    if rm == 'linear':
      problem = OptProblem(data, OptSolverLinear(l2_lam))
    elif rm == 'logistic':
      problem = OptProblem(data, OptSolverLogistic(l2_lam))
    elif rm == 'glm':
      nbr_responses = data.n_responses()
      max_iter = None
      if params.has_key('glm_max_iter'):
        max_iter = params['glm_max_iter']
      if params.has_key('glm_power'):
        glm_power = params['glm_power']
        problem = OptProblem(data, OptSolverGLM(l2_lam, glm_power, 
                                                nbr_responses, max_iter))
      else:
        problem = OptProblem(data, OptSolverGLM(l2_lam, nbr_responses=nbr_responses,
                                                max_iter=max_iter)) 
    else:
      print 'Error: Unknown regression_method - %s' % (rm)
      break

    names = []
    results = []
    times = []
    for om in opt_methods:
      t0 = time.time()
      if om == 'FR':
        results.append(alg_forward(problem, K=K, costs=costs, groups=groups))

      elif om == 'OMP':
        results.append(alg_omp(problem, K=K, costs=costs, groups=groups))

      elif om == 'OMP_stacked':
        results.append(alg_stacked_omp(problem, K=K, costs=costs, groups=groups))

      else:
        print 'Error: Unknown optimization method - %s' % (om)

      times.append(time.time() - t0)
      names.append('%s_%s' % (rm, om))
      print 'Method, time = %s, %f' % (names[-1], times[-1])

    # endfor om 
    ret.append(dict(zip(names, results)))
  # endfor rm 
  return ret 

def regression_fit(X, Y, params, multi_classification=False):
  if multi_classification:
    Y = opt_util.label2indvec(Y)
    print "Y is converted to one-hot"
  if not params.has_key('l2_lam'):
    l2_lam = 1.0 / np.float64(X.shape[0])
  else:
    l2_lam = params['l2_lam']

  rm = 'linear'
  if params.has_key('r_method'):
    rm = params['r_method']

  data = ProblemData(X,Y, l2_lam=l2_lam)
  if rm == 'linear':
    problem = OptProblem(data, OptSolverLinear(l2_lam))
  elif rm == 'logistic':
    problem = OptProblem(data, OptSolverLogistic(l2_lam))
  elif rm == 'glm':
    nbr_responses = data.n_responses()
    max_iter = None
    if params.has_key('glm_max_iter'):
      max_iter = params['glm_max_iter']
    if params.has_key('glm_power'):
      glm_power = params['glm_power']
      problem = OptProblem(data, OptSolverGLM(l2_lam, glm_power, 
        nbr_responses, max_iter))
    else:
      problem = OptProblem(data, OptSolverGLM(l2_lam, 
        nbr_responses=nbr_responses, max_iter=max_iter)) 

  print "Set-up finished "
  # training using all features.
  model, _ = problem.opt_and_score(np.arange(X.shape[1]))

  if rm != 'glm':
    return (rm, model, multi_classification)
  return (rm, (model, problem.solver.calib_funcs), multi_classification)

def regression_predict(X, method_model):
  rm = method_model[0]
  model = method_model[1]
  multi_classification = method_model[2]
  if rm == 'linear':
    Y_hat = OptSolverLinear.predict(X, model)
  elif rm == 'logistic':
    Y_hat = OptSolverLogistic.predict(X, model)
  elif rm == 'glm':
    Y_hat = OptSolverGLM.predict(X, model[0], model[1])

  if multi_classification:
    return opt_util.indvec2label(Y_hat)
  return Y_hat

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
