import opt2
import dataset2
import numpy as np
import scipy


# Streaming anytime training/testing

class StreamProblemData(object): 
    def __init__(self, data_dir, vec_data_fn, costs, groups, l2_lam=1e-6, y_val_func = lambda x:x, call_init=True, compute_YTX=False, load_fn=None):
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
        self.m_X = np.zeros((n_dim,), dtype = np.flat64)
        self.XTX = np.zeros((n_dim, n_dim), dtype = np.float64)
        
        self.n_groups = np.max(self.groups)+1
        self.vec_feats_g = []
        for g in range(n_groups):
            feats_g = np.nonzero(self.groups==g)[0]
            self.vec_feats_g.append(feats_g)

        self.has_init = False
        if call_init:
            self.init(load_fn)

    def init(self, load_fn=None):
        if load_fn is not None:
            fin = np.load(load_fn)
            self.XTX = fin['XTX']
            self.m_X = fin['m_X']
            if self.compute_YTX:
                self.b = fin['b']
            fin.close()
        else:
            if self.compute_YTX:
                self.YTX = np.zeros((self.n_responses, n_dim), dtype=np.float64)
                self.m_Y = np.zeros((self.n_responses), dtype=np.float64)

            for fn_i, data_fn in enumerate(self.vec_data_fn):
                X_i, Y_i = dataset2.load_data(data_fn, self.y_val_func, data_dir=self.data_dir)
                self.n_responses = Y_i.shape[1]
                self.n_X += X_i.shape[0]
                self.m_X += np.sum(X_i, axis=0)
                self.XTX += np.dot(X_i.T, X_i)
                if self.compute_YTX:
                    self.YTX += np.dot(Y_i.T.X_i)
                    self.m_Y += np.sum(Y_i, axis=0)
                
            self.m_X /= self.n_X
            self.XTX /= self.n_X
            if self.compute_YTX:
                self.m_Y /= self.n_X
                self.YTX /= self.n_X
                self.b = self.YTX - self.m_Y.T.dot(self.m_X)

            # TODO name it with time stamp
            np.savez('feature_stats.npz', m_X=self.m_X, b=self.b, XTX=self.XTX)
        
        self.set_l2_lam(self.l2_lam)
        self.has_init = True

    def set_l2_lam(self, l2_lam):
        self.l2_lam = l2_lam

        self.C = self.XTX - np.outer(self.m_X, self.m_X) + np.eye(self.n_dim) * self.l2_lam
        self.group_C_invs = []
        for g, feats_g in enumerate(self.vec_feats_g):
            self.group_C_invs.append(np.linalg.pinv(self.C[feats_g[:, np.newaxis], feats_g]))

    def n_dim(self):
        return self.n_dim
    
    def n_responses(self):
        return self.n_responses

    def load_and_preprocess(self, fn):
        X, Y = dataset2.load_data(fn, self.y_val_func, data_dir=self.data_dir)
        X = X - self.m_X

        # TODO handle Scale of each X dimension ?
        return X, Y

def opt_stream_glm_explicit(vec_data_fn, selected=None, spd, potential_func, mean_func, C_inv, w0=None, intercept=True):
    if selected is None:
        selected = np.arange(spd.n_dim())

    w_len = selected.shape[0]
    if intercept:
        C_inv_tmp = np.zeros((w_len, w_len))
        C_inv_tmp[1: , 1:] = C_inv
        C_inv = C_inv_tmp
        C_inv[0,0] = 1
        w_len += 1

    w = np.zeros((w_len, spd.n_responses())) 
    if w0 is not None:
        w[:w0.shape[0]] = w0

    nbr_iter = 0
    beta = 0.8
    while True:
        L_lipschitz = np.max(abs(w)) * spd.l2_lam + 1

        objective = 0
        delta_w = np.zeros_like(w) 
        n_samples = 0 

        for fn_i, fn in enumerate(vec_data_fn):
            X, Y = spd.load_and_preprocess(fn)
            n_samples += X.shape[0]
            if intercept:
                dot_Xw = X.dot(w[1:]) + w[1]
            else:
                dot_Xw = X.dot(w)
            # TODO overflow/underflow ?
            delta_w += C_inv.dot(X.T.dot(mean_func(dot_Xw) - Y)) / L_lipschitz
            objective += np.sum(potential_func(dot_Xw)) - np.sum(Y * dot_Xw)

        objective /= n_samples
        if intercept:
            objective += spd.l2_lam * np.sum(w[1:] * w[1:]) * 0.5
        else:
            objective += spd.l2_lam * np.sum(w*w) * 0.5

        if nbr_iter > 0:
            conv_num = abs(last_objective - objective) / np.abs(last_objective)
            has_converge = conv_num < 1e-3
            if has_converge:
                break

            if last_objective < objective:
                print "iteration: {}. Step size was too large. Shrinking!!!".format(nbr_iter)
#TODO fix this if this happens. Backtrack for now.
                last_delta_w *= beta
                w = last_w - last_delta_w
                continue

        last_objective = objective

        delta_w /= n_samples
        last_delta_w = spd.l2_lam * w + delta_w
        last_w = w
        w -= last_delta_w 
        
        nbr_iter += 1

    return w, -objective
       
    
class StreamOptSolverGLMExplicit(object):
    def __init__(self, l2_lam=1e-4, intercept=True, mean_func=lambda x:x, potential_func = lambda x: (x**2)*0.5, gradient_func = lambda x:1):
        
        self.mean_func = mean_func
        self.potential_func = potential_func
        self.gradient_func = gradient_func
        
        self.l2_lam = l2_lam
        self.intercept = intercept

    def opt_and_score(self, spd, selected_feats, model0=None):
        w0 = None
        if model0 is not None:
            w0 = model0['w']
        w, score = opt_stream_glm_explicit(spd.vec_data_fn, selected_feats, spd, self.potential_func, self.mean_func, C_inv=np.linalg.pinv(spd.C[selected_feats[:, np.newaxis], selected_feats]), w0=w0, intercept=self.intercept)

        model = {}
        model['intercept'] = self.intercept
        model['w'] = w

        return model, score
        
    def predict(self, spd, fn, model):
        X, Y = spd.load_and_preprocess(fn)
        w = model['w']
        if model['intercept']:
            return self.mean_func(X.dot(w[1:]) + w[0])
        return self.mean_func(X.dot(w))
    
    def init_model(self, n_responses):
        model = {}
        model['intercept'] = self.intercept
        model['w'] = np.zeros((self.intercept, n_responses), dtype=np.float64)
        return model

    def compute_whiten_gradient_norm_unselected(self, spd, selected_feats, selected_groups, model):
        g_offset = {}
        curr_offset = 0
        unselected_feats = []
        for g in range(spd.n_groups):
            if not g in selected_group:
                g_offset[g] = curr_offset
                curr_offset += spd.vec_feats_g[g].shape[0]
                unselected_feats.append(spd.vec_feats_g[g])

        n_unselected = curr_offset
        assert(n_unselected == spd.n_dim - selected_feats.shape[0])
        w_grad_unselected = np.zeros((spd.n_responses,n_unselected), dtype=np.float64)
        unselected_feats = np.hstack(unselected_feats)

        intercept = model['intercept']
        w = model['w']
        n_samples = 0
        for data_fn in spd.vec_data_fn:
            X, Y = spd.load_and_preprocess(data_fn)
            n_samples += X.shape[0]
            if model['intercept']:
                dot_Xw = X[:,selected_feats].dot(w[1:]) + w[0] 
            else:
                dot_Xw = X[:,selected_feats].dot(w)

            # gradient of each sample right before X transpose
            # NxK
            grad_proxy = np.sum((Y - self.mean_func(dot_Xw))[:,np.newaxis,:] * self.gradient_func(dot_Xw), axis=2)

            # apply X transpose and sum up
            # KxD_unselected
            # TODO overflow underflow? 
            w_grad_unselected += grad_proxy.T.dot(X[:, unselected_feats])

        w_grad_unselected /= n_samples

        # compute gradient norm (whitened by group C_inv) for each unselected group.
        w_grad_norm_whiten = {}
        for g,offset in enumerate(g_offset):
            b_g = w_grad_unselected[:, offset:offset+spd.vec_feats_g[g].shape[0]] 
            w_grad_norm_whiten[g] = np.sum(b_g.dot(spd.group_C_invs[g]) * b_g)

        return w_grad_norm_whiten
                

def alg_omp(problem):
    n_groups = problem.data.n_groups
    n_dim = problem.data.n_dim

    mask = np.zeros(n_groups, np.bool)
    selected_groups = np.zeros(K, np.int)
    best_feats = np.zeros(n_dim, np.int)
    best_feats_end = 0
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
        best_feats_end = sel_feats_end

        # warm start vs. cold start
        #last_model, best_score = problem.opt_and_score(best_feats[:best_feats_end], last_model) 
        last_model, best_score = problem.opt_and_score(best_feats[:best_feats_end]) 
        
        c = np.sum(costs[best_groups])
        timestamp = time.time() - t0
        sequence.append((best_score, c, g, best_feats[:best_feats_end], 
                         best_groups, last_model, timestamp))

    return np.asarray(sequence, dtype=[('score', np.float), ('cost', np.float), ('group', np.int),
                      ('selected', object), ('selected_groups', object),
                      ('model', object), ('time', np.float)])



def omp_select_groups(problem, selected_features, mask, model):
    selected_groups = np.where(mask)[0]
    grad_norms = compute_whiten_gradient_norm_unselected(problem.spd, selected_features, selected_groups, model) 

    best_ip = 0.0
    best_g = -1
    for g,norm in enumerate(grad_norms):
        ip = norm / problem.data.costs[g]

        if ip > best_ip:
            best_ip = ip
            best_g = g

    return best_g


class StreamOptProblem(object):
    def __init__(self,stream_data, stream_solver):
        self.spd = stream_data
        self.solver = stream_solver

    def opt_and_score(self, selected_feats, model0):
        return self.solver.opt_and_score(self.spd, selected_feats, model0)

    def init_model(self):
        return self.solver.init_model(self.spd.n_responses())

def stream_regression_fit(vec_X_fn, vec_Y_fn, params, multi_classification=True):
    #TODO
    pass

def stream_regression_predict(vec_X_fn, method_model):
    #TODO
    pass

