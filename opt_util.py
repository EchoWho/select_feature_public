import numpy as np
from scipy.linalg import pinv,inv


# input matrix: [ C, V; U, A]
# use woodbury matrix inversion lemma to invert the input matrix
# return the final inverse
def woodbury_inv(A, U, V, C_inv):
    C_dim = C_inv.shape[0]
    A_dim = A.shape[0]
    if C_dim == 0:
        return pinv(A)
    
    full_inv = np.zeros((C_dim+A_dim, C_dim+A_dim), dtype=np.float64) 
    # return format [ H, -G; -F, E ]
    U_dot_C_inv = U.dot(C_inv)
    C_inv_dot_V = C_inv.dot(V)
    E = inv(A - U_dot_C_inv.dot(V))
    F = E.dot(U_dot_C_inv)
    full_inv[0:C_dim, 0:C_dim] = C_inv_dot_V.dot(F) + C_inv  #H
    full_inv[0:C_dim, C_dim:] = -C_inv_dot_V.dot(E) #-G
    full_inv[C_dim:, 0:C_dim] = -F
    full_inv[C_dim:, C_dim:] = E
    return full_inv


def label2indvec2(Y, nbr_classes):
    pos_samples = np.where(Y>=0)[0].ravel()
    indvec = np.zeros((Y.shape[0], nbr_classes))
    indvec[(pos_samples, Y.ravel().astype('int32')[pos_samples])] = 1
    return indvec

def label2indvec(Y, nbr_classes=None):
  if nbr_classes == None:
    nbr_classes = np.unique(Y).shape[0]
  indvec = np.zeros((Y.shape[0], nbr_classes))
  indvec[(range(Y.shape[0]), Y.ravel().astype('int32'))] = 1
  return indvec

def indvec2label(indvec):
  return np.argmax(indvec, axis=1)[:, np.newaxis]

def dcg(Y_hat, Y, top=5):
  vals = Y[sorted(range(len(Y_hat)), key=lambda x: Y_hat[x])[-1:-top-1:-1]]
  
  list_zeros = np.where(vals == 0)[0]
  if len(list_zeros > 0):
    #print "setting 0 from indx: {}".format(len(list_zeros))
    vals[list_zeros[0]:] = 0

  weights = np.log2(np.arange(vals.shape[0]) + 1) 
  weights[0] = 1
  return np.sum(vals.ravel() / weights )

def ndcg(Y_hat, Y, top=5, idcg_stored=None):
  if idcg_stored != None:
    return dcg(Y_hat, Y, top) / idcg_stored
  val1 = dcg(Y_hat, Y, top)
  val2 = dcg(Y, Y, top)
  if (val2 == 0):
    return 1.0
  return val1 / val2

def ndcg_overall(Y_hat, Y, query_starts, top=5):
  L = len(query_starts) - 1
  return  np.mean([ndcg(Y_hat[query_starts[i]:query_starts[i+1]], 
                        Y[query_starts[i]:query_starts[i+1]], top)  for i in range(L)])
    

def generate_chunks(N, k, complement=True):
  a = list(range(N))
  chunk_size = N / k
  chunks =  np.array(zip(*[iter(a)]*chunk_size))
  
  c_chunks = np.zeros((k, chunk_size * (k-1)), dtype=np.int64)
  for i in range(k):
    c_chunks[i, :] = np.hstack([ chunk for ci, chunk in enumerate(chunks) if ci != i])
  return chunks, c_chunks
