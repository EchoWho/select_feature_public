import opt2
import numpy as np
import sklearn.linear_model

def doit():
  print "start loading"
  d_mnist = np.load('mnist_data/mnist.npz')
  Y = d_mnist['Y']
  Y_test = d_mnist['Ytest']

  d_train = np.load('mnist_data/X_phi_train_mpp_15_theta_2_gamma_0.5_7-17-18-10.npz')
  X = d_train['arr_0']
  m_X = np.mean(X, axis=0)
  std_X = np.std(X, axis=0)
  std_X += std_X==0
  X = (X - m_X) / std_X

  d_test = np.load('mnist_data/X_phi_test_mpp_15_theta_2_gamma_0.5_7-17-18-11.npz')
  X_test = d_test['arr_0']
  X_test = (X_test - m_X ) / std_X

  print "finsihed loading"
  
  lr = sklearn.linear_model.LogisticRegression(C=1.666667)
  #lr.fit(X, Y.ravel())
  #print "MAP : %f" %( lr.score(X_test, Y_test.ravel()))

  params = {}
  params['l2_lam'] = 1e-5
  params['r_method'] = 'logistic' # 'linear', 'glm'
  params['glm_max_iter'] = 5
  params['glm_power'] = 5
  fitted_model = opt2.regression_fit(X,Y, params, multi_classification=True)

  Y_hat = opt2.regression_predict(X_test, fitted_model)

  err = np.sum(Y_hat != Y_test) / np.float64(Y_test.shape[0])
  print err

if __name__ == "__main__":
  doit()
