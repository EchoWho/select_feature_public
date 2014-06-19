import numpy as np
import sklearn.linear_model
import sys
import dataset 

#if __name__ == "__main__":

def doit():

  D = dataset.Dataset('vision_data', 'vision_results', 'groups.mat')
  fn_trains = sys.argv[1]#'fold11.train.txt'
  params = {}
  params['l2_lam'] = 1e-5
  params['do_FR'] = False
  params['logistic'] = True

  do_sklearn_logistic = False
  if do_sklearn_logistic:
    X_tra, Y_tra, b, C_no_regul, m_X, m_Y, std_X = D.pretrain(fn_trains)
    model = np.load(D.filename_model(fn_trains, params))
    nbr_grps = 3
    selected_groups = set(model['OMP']['group'][1:(nbr_grps+1)])
    selected_features = np.nonzero([ (v in selected_groups) for _, v in enumerate(D.groups)])[0]

    X_tra, Y_tra, _, _, _, m_Y, _ = D.pretrain(fn_trains)
    print X_tra.shape

    lr = sklearn.linear_model.LogisticRegression()
    #lr = sklearn.linear_model.LinearRegression(fit_intercept=False)
    lr.fit(X_tra[:, selected_features],Y_tra > 0)

    fn_test = 'fold4.test.txt'
    X_tes, Y_tes, _, _, _, _, _ = D.pretrain(fn_test)

    model_omp = model['OMP']
    selected_X = model_omp['selected'][nbr_grps]
    w = model_omp['w'][nbr_grps]

    Y_hat = lr.predict(X_tes[:, selected_features])

    print "Guess normal : %f" % ( np.sum(Y_tes + m_Y <= 0.5)  / np.float64(Y_tes.shape[0]))
    print "MAP: %f" % ( np.sum((Y_hat > 0.5) ==  (Y_tes > 0)) * 1.0 / Y_tes.shape[0] )

    #print "MAP: %f" % ( np.sum((np.dot(X_tes[:, selected_X], w) + m_Y > 0.5) ==  (Y_tes + m_Y > 0.5)) * 1.0 / Y_tes.shape[0] )
    return X_tra, Y_tra, X_tes, Y_tes,  Y_hat, lr
  else:
    X_tra, Y_tra, b, C_no_regul, m_X, m_Y, std_X = D.pretrain(fn_trains)
    D.train(fn_trains, params)
    fn_test = sys.argv[2] #'ca11_no_label_bov_XY.mat'
    return D.compute_budget_vs_loss_multi_files(fn_test, fn_trains, params)
    

if __name__ == "__main__":
  doit()
