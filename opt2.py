import numpy as np
import sys
import functools

def logistic_mean_func(dot_Xw):
  exp_Xw = np.exp(-dot_Xw)
  return 1.0 / (1.0 + exp_Xw)

def logistic_gradient(dot_Xw):
  exp_Xw = np.exp(-dot_Xw)
  return exp_Xw / (1.0 + exp_Xw)**2

def logistic_potential(dot_Xw):
  return np.log(1.0 + np.exp(dot_Xw))

def logistic_lipschitz():
  return 1

def opt_linear_bC(b, C):
  return np.linalg.pinv(C).dot(b)

def opt_linear(X,Y,C_inv=None):
  b = X.T.dot(Y) / np.float64(Y.shape[0])
  if C_inv == None:
    C = X.T.dot(X) / np.float64(Y.shape[0])
    C = (C + C.T) / 2.0
    C_inv = np.linalg.pinv(C)
  return C_inv.dot(b)
  
def opt_glm_explicit(X, Y, potential_func, mean_func, mean_lipschitz, w0=None,
                     intercept=True, C_inv=None, l2_lam=None):
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
    C_inv = np.linalg.pinv(C)

  if intercept:
    C_inv_tmp = np.zeros((w_len, w_len))
    C_inv_tmp[1: , 1:] = C_inv
    C_inv = C_inv_tmp
    C_inv[0,0] = 1
    X = np.hstack([np.ones((nbr_samples, 1)), X])

  w = np.zeros((w_len, nbr_responses))
  if w0 != None:
    w[:w0.shape[0]] = w0
  
  has_converge = False
  is_first = True
  while not has_converge:
    dot_Xw = X.dot(w)
    residual = ( mean_func(dot_Xw) - Y ) / nbr_samples 
  
    objective = (np.sum(potential_func(dot_Xw)) - np.sum(Y * dot_Xw)) / nbr_samples
    if intercept:
      objective += l2_lam * np.sum( (w[1:] * w[1:]) ) / 2.0
    else:
      objective += l2_lam * np.sum( (w * w) ) / 2.0
    if is_first:
      is_first = False
    else:
      has_converge = abs(last_objective - objective) / np.abs(last_objective) < 1e-5
      if has_converge:
        break
    last_objective = objective

    delta_w = (1 / mean_lipschitz) * C_inv.dot(X.T.dot(residual))
    regul_delta_w = w * l2_lam
    if intercept:
      regul_delta_w[0] = 0
    w -= delta_w + regul_delta_w

  return w, objective 
      
def opt_glm_implicit(X, Y, calib_funcs, w0=None, 
                     intercept=True, C_inv=None, l2_lam=None)
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
  if w0 != None:
    w[:w0.shape[0]] = w0
    Y_hat = X.dot(w)
  else:
    Y_hat = np.zeros(Y.shape)
  vec_w.append(w)

  has_converge = False
  is_first = True
  while not has_converge:
    w = opt_linear(X, Y-Y_hat, C_inv)
    vec_w.append(w)
    Y_tilt = Y_hat + X.dot(w)
    G_Y_tilt = np.array([ f(Y_tilt).ravel() for f in calib_funcs ])
    psuedo_fubini = G_Y_tilt.dot(G_Y_tilt.T)
    psuedo_fubini_Y = G_Y_tilt.dot(Y.ravel())
    w_tilt = np.linalg.pinv(psuedo_fubini).dot(psuedo_fubini_Y)
    vec_w_tilt.append(w_tilt) 
    Y_hat = G_Y_tilt.T.dot(w_tilt).T.reshape(Y.shape)
    if is_first:
      is_first = False
    elif np.norm(Y_hat - old_Y_hat) / old_norm < 1e-5:
      has_converge = True
      break
    old_Y_hat = Y_hat
    old_norm = np.norm(Y_hat)
  return vec_w, vec_w_tilt, Y_hat

def glm_predict(X, vec_w, vec_w_tilt, calib_funcs, intercept=True, 
                compute_grad=False, Y=None, w_tilt_grad_G=None):
  T = len(vec_w) - 1
  nbr_responses = vec_w[0].shape[1]
  nbr_samples = X.shape[0]
  nbr_feats = X.shape[1]
  if intercept:
    X = np.hstack([np.ones((nbr_samples, 1)), X])
  if compute_grad:
    vec_Y_tilt = np.zeros((T, nbr_samples, nbr_responses))
  Y_hat = X.dot(vec_w[0])
  for t in range(T-1):
    Y_hat += X.dot(vec_w[t+1]) #Y_tilt
    if compute_grad:
      vec_Y_tilt[t] = Y_hat # Y_tilt recorded
    Y_hat = np.array([ f(Y_hat) for f in calib_funcs ]).T.dot(vec_w_tilt[t]).T
  if compute_grad:
    if Y == None:
      sys.exit(1)
    U = Y_hat - Y
    t = T - 1
    grad_norm_square = np.zeros(nbr_feats)
    while t > 0:
      U = w_tilt_grad_G(Y_tilt=vec_Y_tilt[t], w_tilt=vec_w_tilt[t], U)
      # Use U^t
      for d in range(nbr_feats):
        grad_norm_square[d] += np.sum((U.T.dot(X)) ** 2 , axis=0)
    return Y_hat, grad_norm_square  
  return Y_hat

def generate_glm_funcs(K, P):
  nbr_classes = K
  calib_funcs = []
  gradients = []
  if K == 1:
    for p in range(P):
      if p == 0:
        calib_funcs.append(lambda X : np.ones(X.shape[0]))
      else:
        calib_funcs.append(lambda X : X**p)
  else:
    def k_th_column_matrix(col_vec, k, K):
      z = np.zeros((col_vec.shape[0], K))
      z[:,k] = col_vec
      return z
    calib_funcs.append(lambda X : X)
    for k in range(K):
      for p in range(P):
        calib_funcs.append(lambda X : k_th_column_matrix(X[:,k]**p, k, X.shape[1]))
  return calib_funcs, w_tilt_grad_G(K,P)

def w_tilt_grad_G(K, P):
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
