import numpy as np
import scipy


class BradleyLoader(object):
    def __init__(self):
        self.keys = ['acf_warped_seconds',\
                 'icf_warped',\
                 'warped_cnn_fv6',\
                 'boxes',\
                 'warped_cnn_seconds',\
                 'acf_warped',\
                 'icf_warped_seconds',\
                 'warped_cnn_predictions']

        self.feat_keys = ['icf_warped', 'acf_warped', 'warped_cnn_fv6', 'warped_cnn_predictions'] 

    def load_data(self, fn, y_val_func=lambda x:x, data_dir='.', load_for_train=False, neg_threshold=0.3, pos_threshold=0.5):
        fin = np.load('{}/{}'.format(data_dir, fn))
        X = np.hstack([fin[feat_key] for feat_key in self.feat_keys])

        #print "Warning loading only the first 100 dim for testing"
        #X = X[:, :100]

        #boxes:
        # 0    1    2    3    4     5        6       
        # xmin ymin xmax ymax score image_id max_iou 
        # 7             8        9 
        # max_iou_class max_base max_base_class 
        iou = fin['boxes'][:, 6]
        fin.close()

        # 0 : iou < 0.3
        # 1 : iou >=0.5 
        # 0.5 : iou in between
        #neg_idx = np.where(iou < neg_threshold)[0]
        #pos_idx = np.where(iou >= pos_threshold)[0]
        #Y = np.ones((X.shape[0], 1)) * 0.5
        #Y[neg_idx] = 0
        #Y[pos_idx] = 1

        Y = iou * 2.5 - 0.75
        Y = Y[:, np.newaxis]
        neg_idx = np.where(Y <0)[0]
        pos_idx = np.where(Y >=1)[0]
        Y[neg_idx] = 0
        Y[pos_idx] = 1
        load_for_train = False

        # select only pos and neg samples 
        if load_for_train:
            sample_idx = np.hstack([neg_idx, pos_idx])
            X = X[sample_idx]
            Y = Y[sample_idx]

        return X, y_val_func(Y)
