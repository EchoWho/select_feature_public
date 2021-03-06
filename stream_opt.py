import pdb
import numpy as np
import os.path
import time
from scipy.linalg import pinv,inv
from opt_util import woodbury_inv
from copy import deepcopy

#
# Class Hierarchy:
#  
#  alg_omp requires a Problem,
#  Problem requires two parts: ProblemData, ProblemSolver
#
#  ProblemData: preprocess (m_X, m_Y, XTX,XTY, and etc); load_fn ; group structures, costs
#  ProblemSolver: Linear Regression / Logistic regression (GLM)
#
#  Misc: Use data/solver only, regression_fit/regression_predict. By-pass alg_omp
#

##########################
# Core algorithm: OMP
##########################

def alg_omp(problem, save_steps=False, step_fn_prefix='step_result', test_all_feats=False):
    problem.init_cost_manager()
    n_groups = problem.data.n_groups
    n_dim = problem.data.n_dim

    mask = np.zeros(n_groups, np.bool)
    selected_groups = np.zeros(n_groups, np.int)
    best_feats = np.zeros(n_dim, np.int)
    best_feats_end = 0
    best_score = 0
    last_C_inv = np.eye(0, dtype=np.float64)
    last_model=problem.init_model()
    # score, cost, group, selected, selected_groups, model, time
    sequence = [(0.0, 0.0, -1, [], [], last_model, 0)]

    t0 = time.time()
    for k in range(n_groups):
        print 'OMP Iteration %d' % k
        g = omp_select_groups(problem, best_feats[:best_feats_end], mask, last_model)

        if g == -1:
            print 'Exited with no group selected on iteration %d' % (k+1)
            break

        selected_groups[k] = g
        mask[g] = True

        best_groups = selected_groups[:k+1]

        feats_g = problem.data.vec_feats_g[g]
        sel_feats_end = best_feats_end + feats_g.shape[0]
        best_feats[best_feats_end:sel_feats_end] = feats_g
        last_bfe = best_feats_end
        best_feats_end = sel_feats_end

        fix_previous_weights=False
        if not fix_previous_weights:
            # warmstart
            #last_model, best_score = problem.opt_and_score(best_feats[:best_feats_end], last_model) 

            # Woodbury. Remember that the performancemay decrease due to bad inverses. so we need to check and recompute inverse
            last_model, score, last_C_inv = problem.opt_and_score_woodbury(best_feats[:best_feats_end], last_C_inv=last_C_inv, 
                                                                          last_sel=best_feats[:last_bfe], new_sel=feats_g) 
            # shouldn't have drop in performance;
            if score < best_score:
                print "woodbury failure .... recompute model"
                last_C_inv = inv(problem.data.C[best_feats[:best_feats_end, np.newaxis], best_feats[:best_feats_end]])
                last_model, score = problem.opt_and_score(best_feats[:best_feats_end], C_inv = last_C_inv)
        
        else: 
            # Previous weights are kept fixed
            if k==0:
                last_model=None
            last_model, score = problem.opt_and_score_additive(last_sel=best_feats[:last_bfe],new_sel=feats_g,model0=last_model)

        best_score = score
        print "best_score {}".format(best_score)
        
        #c = np.sum(problem.data.costs[best_groups])
        problem.costs.choose(g)
        c = problem.costs.total_cost()

        timestamp = time.time() - t0
        sequence.append((best_score, c, g, best_feats[:best_feats_end], 
                         best_groups, last_model, timestamp))

        if save_steps:
            step_result = np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                      ('selected', object), ('selected_groups', object),
                      ('model', object), ('time', np.float)])
            np.savez('{}_{}.npz'.format(step_fn_prefix, k), step_result=step_result)

    # report result for all feats. 
    if test_all_feats:
        print "compute score will all features"
        last_model, best_score = problem.opt_and_score(np.arange(n_dim))
        c = problem.costs.total_possible_cost()
        timestamp = time.time() - t0
        sequence.append((best_score, c, -1, np.arange(n_dim), 
                         np.arange(n_groups), last_model, timestamp))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                      ('selected', object), ('selected_groups', object),
                      ('model', object), ('time', np.float)])



def omp_select_groups(problem, selected_feats, mask, model):
    grad_norms = problem.solver.compute_whiten_gradient_norm_unselected(problem.data, selected_feats, mask, model) 

    best_ip = -np.inf
    best_g = -1
    for _,g in enumerate(grad_norms):
        cost_g = problem.costs.cost_of(g)
        if cost_g != 0:
            ip = grad_norms[g] / cost_g
        else:
            best_g = g
            break

        if ip > best_ip:
            best_ip = ip
            best_g = g
    
    print "best ip/cost {} from group {}".format(best_ip, best_g)
    return best_g


######################################
# Problem, Problem Data
######################################

class StreamOptProblem(object):
    def __init__(self,stream_data, stream_solver):
        self.data = stream_data
        self.solver = stream_solver

    def opt_and_score_additive(self, last_sel, new_sel, model0):
        return self.solver.opt_and_score_additive(self.data, last_sel, new_sel, model0) 

    def opt_and_score_woodbury(self, selected_feats, last_C_inv, last_sel, new_sel, model0=None):
        C = self.data.C[selected_feats[:, np.newaxis], selected_feats]
        U = self.data.C[new_sel[:,np.newaxis],last_sel]
        C_inv = woodbury_inv(A=self.data.C[new_sel[:,np.newaxis], new_sel], 
                             U=U,
                             V=U.T,
                             C_inv=last_C_inv)
        model, score = self.solver.opt_and_score(self.data, selected_feats, model0, C_inv)
        return model, score, C_inv

    def opt_and_score(self, selected_feats, model0=None, C_inv=None):
        return self.solver.opt_and_score(self.data, selected_feats, model0, C_inv)

    def init_model(self):
        return self.solver.init_model(self.data.n_responses)

    def init_cost_manager(self):
        self.costs = deepcopy(self.data.costs)


class StreamProblemData(object): 
    def __init__(self, n_responses, loader, data_dir, vec_data_fn, costs, groups, l2_lam=1e-6, y_val_func = lambda x:x, call_init=True, compute_XTY=False, load_stats=False, load_dir=None):

        self.loader = loader

        # compute class C, and over all C. 
        self.data_dir = data_dir
        self.y_val_func = y_val_func
        self.vec_data_fn = vec_data_fn
        self.costs = costs
        self.groups = groups
        self.l2_lam = l2_lam
        
        n_dim = groups.shape[0]
        self.n_dim = n_dim
        self.n_X = 0
        self.m_X = np.zeros((n_dim,), dtype = np.float64)
        self.std_X = np.zeros((n_dim,), dtype=np.float64)
        self.XTX = np.zeros((n_dim, n_dim), dtype = np.float64)

        self.n_responses = n_responses
        self.m_Y = np.zeros((self.n_responses), dtype=np.float64)
        self.YTY = np.zeros((n_responses, n_responses), dtype=np.float64)
        
        self.n_groups = np.max(self.groups)+1
        self.vec_feats_g = []
        for g in range(self.n_groups):
            feats_g = np.nonzero(self.groups==g)[0]
            self.vec_feats_g.append(feats_g)

        self.compute_XTY = compute_XTY

        self.has_init = False
        if call_init:
            self.init(load_stats, load_dir)

    def init(self, load_stats=False, load_dir=None):
        means_fn = '{}/feature_means.npz'.format(load_dir)
        stats_fn = '{}/feature_stats.npz'.format(load_dir)
        group_C_invs_fn = '{}/group_C_invs.npz'.format(load_dir)

        if load_stats and (load_dir is not None):
            print "load data statistics from {} ...".format(stats_fn)
            fin = np.load(stats_fn)
            self.XTX = fin['XTX']
            self.YTY = fin['YTY']
            self.m_X = fin['m_X']
            self.std_X = fin['std_X']
            self.std_X_inv = 1.0 / self.std_X
            self.m_Y = fin['m_Y']
            if self.compute_XTY:
                self.b = fin['b']
            fin.close()
            print "done"

        else:
            if self.compute_XTY:
                self.XTY = np.zeros((self.n_dim, self.n_responses), dtype=np.float64)

            load_means = False
            if load_means:
                fin = np.load(means_fn)
                self.n_X = fin['n_X']
                self.m_X = fin['m_X']
                self.std_X = fin['std_X']
                self.std_X_inv = 1.0 / self.std_X
                self.m_Y = fin['m_Y']
                fin.close()
            else:
                for fn_i, data_fn in enumerate(self.vec_data_fn):
                    print 'means- Loading file {} : {}'.format(fn_i, data_fn)
                    X_i, Y_i = self.loader.load_data(data_fn, self.y_val_func, data_dir=self.data_dir, load_for_train=True)
                    self.n_responses = Y_i.shape[1]
                    self.n_X += X_i.shape[0]
                    self.m_X += np.sum(X_i, axis=0)
                    self.m_Y += np.sum(Y_i, axis=0)
                self.m_X /= self.n_X
                self.m_Y /= self.n_X
                
                for fn_i, data_fn in enumerate(self.vec_data_fn):
                    print 'std - Loading file {} : {}'.format(fn_i, data_fn)
                    X_i, Y_i = self.loader.load_data(data_fn, self.y_val_func, data_dir=self.data_dir, load_for_train=True)
                    self.std_X += np.sum((X_i - self.m_X)**2, axis=0)
                self.std_X = np.sqrt(self.std_X / self.n_X)
                self.std_X += self.std_X == 0

                self.std_X_inv = 1.0 / self.std_X
                 
                np.savez(means_fn, n_X=self.n_X, m_X=self.m_X, m_Y=self.m_Y, std_X=self.std_X)

            self.n_X = 0
            for fn_i, data_fn in enumerate(self.vec_data_fn):
                print 'XTX - Loading file {} : {}'.format(fn_i, data_fn)
                X_i, Y_i = self.load_and_preprocess(data_fn, load_for_train=True)
                self.n_X += X_i.shape[0]
                # Overflow Underflow TODO
                self.XTX += np.dot(X_i.T, X_i)
                self.YTY += np.dot(Y_i.T, Y_i)
                if self.compute_XTY:
                    self.XTY += np.dot(X_i.T, Y_i - self.m_Y)
                
            print 'Compute m_X, XTX'
            self.XTX = (self.XTX.T / 2.0 + self.XTX / 2.0) / self.n_X
            self.YTY = (self.YTY.T / 2.0 + self.YTY / 2.0) / self.n_X
            if self.compute_XTY:
                self.XTY /= self.n_X
                self.b = self.XTY # - np.outer(self.m_X, self.m_Y)

            np.savez(stats_fn, m_X=self.m_X, m_Y=self.m_Y, std_X=self.std_X, b=self.b, XTX=self.XTX, YTY=self.YTY)
        
        if load_stats and os.path.isfile(group_C_invs_fn):
            fin = np.load(group_C_invs_fn)
            self.l2_lam = fin['l2_lam']
            self.C = fin['C']
            self.C = (self.C + self.C.T) / 2.0
            self.group_C_invs = fin['group_C_invs']
            #for g, g_C_inv in enumerate(self.group_C_invs):
            #    self.group_C_invs[g] = (g_C_inv + g_C_inv.T) /2.0
            fin.close()
        else:
            self.set_l2_lam(self.l2_lam)
            np.savez(group_C_invs_fn, group_C_invs=self.group_C_invs, C=self.C, l2_lam=self.l2_lam)
        self.has_init = True

    def set_l2_lam(self, l2_lam):
        print "Set l2_lam and compute group covariance inverse..."
        self.l2_lam = l2_lam

        self.C = self.XTX + np.eye(self.n_dim) * self.l2_lam
        self.C = self.C.T /2.0 + self.C /2.0
        self.group_C_invs = []
        for g, feats_g in enumerate(self.vec_feats_g):
            print 'Compute group inv {} of dim {}'.format(g, feats_g.shape[0])
            self.group_C_invs.append(inv(self.C[feats_g[:, np.newaxis], feats_g]))
        print "done"

    def load_and_preprocess(self, fn, load_for_train=False):
        X, Y = self.loader.load_data(fn, self.y_val_func, self.data_dir, load_for_train)
        X = (X - self.m_X) * self.std_X_inv

        # TODO handle Scale of each X dimension ?
        return X, Y


######################################
# Solvers
######################################


def opt_stream_glm_explicit(vec_data_fn, spd, potential_func, mean_func, C_inv, selected=None, w0=None, intercept=True):
    if selected is None:
        selected = np.arange(spd.n_dim)

    w_len = selected.shape[0] + int(intercept)

    w = np.zeros((w_len, spd.n_responses)) 
    if w0 is not None:
        w[:w0.shape[0]] = w0

    nbr_iter = 0
    beta = 0.8
    while True:
        L_lipschitz = np.max(abs(w)) * spd.l2_lam + 1

        objective = 0
        delta_w = np.zeros_like(w) 
        n_samples = 0 
        avg_res = 0

        for fn_i, fn in enumerate(vec_data_fn):
            X, Y = spd.load_and_preprocess(fn, load_for_train=True)
            X = X[:, selected]
            n_samples += X.shape[0]
            # TODO overflow/underflow ?
            if intercept:
                dot_Xw = X.dot(w[1:]) + w[0]
                res = mean_func(dot_Xw) - Y
                delta_w[1:] += C_inv.dot(X.T.dot(res))
                delta_w[0] += np.sum(res, axis=0)
            else:
                dot_Xw = X.dot(w)
                res = mean_func(dot_Xw) - Y
                delta_w += C_inv.dot(X.T.dot()) / L_lipschitz
            objective += np.sum(potential_func(dot_Xw)) - np.sum(Y * dot_Xw)
            avg_res += np.sum(res, axis=0)

        #pdb.set_trace()

        objective /= n_samples
        avg_res /= n_samples
        #if intercept:
        #    objective += spd.l2_lam * np.sum(w[1:] * w[1:]) * 0.5
        #else:
        #    objective += spd.l2_lam * np.sum(w*w) * 0.5
        objective += spd.l2_lam * np.sum(w*w) * 0.5

        #print "iter {}, objective {}, residual {}".format(nbr_iter, objective, avg_res)
        if objective == np.inf:
            pdb.set_trace()

        if nbr_iter > 0:
            conv_num = abs(last_objective - objective) / np.abs(last_objective)
            has_converge = conv_num < 1e-3
            if has_converge:
                break

            if last_objective < objective:
                print "iteration: {}. Shrinking!!! objective {}, step_size {}".format(nbr_iter, objective, np.linalg.norm(last_delta_w))
                #TODO fix this if this happens. Backtrack for now.
                last_delta_w *= beta
                w = last_w - last_delta_w
                continue

        last_objective = objective

        delta_w /= n_samples
        last_delta_w = spd.l2_lam * w + delta_w
        last_w = w
        w -= last_delta_w 
        #print "delta_w_size {}, grad_size {}".format(np.linalg.norm(delta_w), np.linalg.norm(last_delta_w))
        
        nbr_iter += 1

    return w, -objective

class StreamOptSolverLinear(object):
    def __init__(self, l2_lam=1e-4, intercept=True):
        self.l2_lam = l2_lam
        self.intercept = intercept

    def opt_and_score_additive(self, spd, last_sel, new_sel, model0=None):
        model = {}
        model['intercept'] = self.intercept
        if self.intercept:
            model['m_Y'] = spd.m_Y

        b = spd.b[new_sel]
        if model0 is not None:
            b -= np.dot(spd.C[new_sel[:,np.newaxis],last_sel], model0['w'])
        C = spd.C[new_sel[:,np.newaxis],new_sel]
        w = inv(C).dot(b)
    
        if model0 is not None:
            model['w'] = np.vstack((model0['w'], w))
        else:
            model['w'] = w

        score = np.sum((2 * b - C.dot(w))*w - self.l2_lam*w*w )
        return model, score

    def opt_and_score_woodbury(self, spd, selected_feats, last_C_inv, last_sel, new_sel, model0=None):
        model = {}
        model['intercept'] = self.intercept
        if self.intercept:
            model['m_Y'] = spd.m_Y

        b = spd.b[selected_feats]
        C = spd.C[selected_feats[:, np.newaxis], selected_feats]
        U = spd.C[new_sel[:,np.newaxis],last_sel]
        C_inv = woodbury_inv(A=spd.C[new_sel[:,np.newaxis], new_sel], 
                             U=U,
                             V=U.T,
                             C_inv=last_C_inv)
        #C_inv = (C_inv + C_inv.T) / 2.0
        w = C_inv.dot(b)
        model['w'] = w
        score = np.sum((2 * b - C.dot(w))*w - self.l2_lam*w*w )
        return model, score, C_inv

    def opt_and_score(self, spd, selected_feats, model0=None, C_inv=None):
        model = {}
        model['intercept'] = self.intercept
        if self.intercept:
            model['m_Y'] = spd.m_Y

        b = spd.b[selected_feats] 
        C = spd.C[selected_feats[:, np.newaxis], selected_feats]
        if C_inv is None:
            w = inv(C).dot(b)
        else:
            w = C_inv.dot(b)
        model['w'] = w

        score = np.sum((2 * b - C.dot(w))*w - self.l2_lam*w*w )
        return model,score

    def predict(self, spd, fn, model, selected_feats=None):
        X, Y = spd.load_and_preprocess(fn) 
        if selected_feats is not None:
            X = X[:, selected_feats]
        Y_hat = X.dot(model['w'])
        if model['intercept']:
            Y_hat += model['m_Y'] 
        return Y_hat, Y

    def init_model(self, n_responses):
        model={}
        model['intercept'] = self.intercept
        if self.intercept:
            model['m_Y'] = np.zeros(n_responses, dtype=np.float64)
        model['w'] = np.zeros((0, n_responses), dtype=np.float64)
        return model

    def compute_whiten_gradient_norm_unselected(self, spd, selected_feats, mask, model):
        unselected_groups = np.where(~mask)[0]
        unselected_feats = np.hstack([ spd.vec_feats_g[g] for g in unselected_groups ])

        b = spd.b[unselected_feats]  # D_unsel x K
        C = spd.C[unselected_feats[:, np.newaxis], selected_feats] #D_unsel x D_sel

        w = model['w'] #D_sel x K
        proxy = (b - C.dot(w)) # k x D_unsel
        
        # You might think that this is necessary but it is NOT!
        # b already took care of setting m_Y =0
        if model['intercept']: 
            proxy -= model['m_Y']
        proxy = proxy.T

        w_grad_norm_whiten = {}
        f = 0 #front
        for g in unselected_groups:
            g_len = spd.vec_feats_g[g].shape[0]
            proxy_g = proxy[:, f:f+g_len] # K x D_unsel_g
            w_grad_norm_whiten[g] = np.sum(proxy_g.dot(spd.group_C_invs[g])*proxy_g)
            f += g_len

        #print w_grad_norm_whiten
        return w_grad_norm_whiten
       
    
class StreamOptSolverGLMExplicit(object):
    def __init__(self, l2_lam=1e-4, intercept=True, mean_func=lambda x:x, potential_func = lambda x: (x**2)*0.5, gradient_func = lambda x:1):
        self.mean_func = mean_func
        self.potential_func = potential_func
        self.gradient_func = gradient_func
        
        self.l2_lam = l2_lam
        self.intercept = intercept

    def opt_and_score(self, spd, selected_feats, model0=None, C_inv=None):
        w0 = None
        if model0 is not None:
            w0 = model0['w']

        if C_inv is None:
            C_inv = inv(spd.C[selected_feats[:, np.newaxis], selected_feats])
        w, score = opt_stream_glm_explicit(spd.vec_data_fn, spd, self.potential_func, self.mean_func, C_inv, selected_feats, w0=w0, intercept=self.intercept)

        model = {}
        model['intercept'] = self.intercept
        model['w'] = w
        return model, score
        
    def predict(self, spd, fn, model, selected_feats=None):
        X, Y = spd.load_and_preprocess(fn)
        if selected_feats is not None:
            X = X[:, selected_feats]
        w = model['w']
        if model['intercept']:
            return self.mean_func(X.dot(w[1:]) + w[0])
        return self.mean_func(X.dot(w)), Y
    
    def init_model(self, n_responses):
        model = {}
        model['intercept'] = self.intercept
        model['w'] = np.zeros((self.intercept, n_responses), dtype=np.float64)
        return model

    def compute_whiten_gradient_norm_unselected(self, spd, selected_feats, mask, model):
        unselected_groups = np.where(~mask)[0]
        unselected_feats = np.hstack([ spd.vec_feats_g[g] for g in unselected_groups ])

        intercept = model['intercept']
        w = model['w']
        n_samples = 0
        w_grad_unselected = np.zeros((w.shape[1], unselected_feats.shape[0]), np.float64)
        for data_fn in spd.vec_data_fn:
            X, Y = spd.load_and_preprocess(data_fn, load_for_train=True)
            n_samples += X.shape[0]
            if model['intercept']:
                dot_Xw = X[:,selected_feats].dot(w[1:]) + w[0] 
            else:
                dot_Xw = X[:,selected_feats].dot(w)

            # gradient of each sample right before X transpose
            # NxK
            grad_proxy = np.sum((Y - self.mean_func(dot_Xw))[:,np.newaxis,:] * self.gradient_func(dot_Xw), axis=2)

            #print grad_proxy.shape
            #print X[:,unselected_feats].shape

            # apply X transpose and sum up
            # KxD_unselected
            # TODO overflow underflow? 
            w_grad_unselected += grad_proxy.T.dot(X[:, unselected_feats])

        w_grad_unselected /= n_samples

        # compute gradient norm (whitened by group C_inv) for each unselected group.
        w_grad_norm_whiten = {}
        f = 0
        for g in unselected_groups:
            g_len = spd.vec_feats_g[g].shape[0]
            b_g = w_grad_unselected[:, f:f+g_len]
            w_grad_norm_whiten[g] = np.sum(b_g.dot(spd.group_C_invs[g]) * b_g)
            f += g_len
        return w_grad_norm_whiten
                

######################################
# Use only the solvers. 
######################################

def stream_regression_fit(vec_X_fn, vec_Y_fn, params, multi_classification=True):
    #TODO
    pass

def stream_regression_predict(vec_X_fn, method_model):
    #TODO
    pass

