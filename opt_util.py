import numpy as np

def label2indvec(Y, nbr_classes=None):
  if nbr_classes == None:
    nbr_classes = np.unique(Y).shape[0]
  indvec = np.zeros((Y.shape[0], nbr_classes))
  indvec[(range(Y.shape[0]), Y.ravel().astype('int32'))] = 1
  return indvec

def indvec2label(indvec):
  return np.argmax(indvec, axis=1)[:, np.newaxis]
