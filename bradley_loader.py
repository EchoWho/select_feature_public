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

        self.feat_keys = ['icf_warped', 'acf_warped', 'warped_cnn_predictions', 'warped_cnn_fv6'] 

    def load_data(self, fn, y_val_func=lambda x:x, data_dir='.', load_for_train=False, neg_threshold=0.3, pos_threshold=0.5):
        fin = np.load('{}/{}'.format(data_dir, fn))
        X = np.hstack([fin[feat_key] for feat_key in self.feat_keys])

        #boxes:
        # 0    1    2    3    4     5        6       7             8        9 
        # xmin ymin xmax ymax score image_id max_iou max_iou_class max_base max_base_class 
        iou = fin['boxes'][:, 6]
        Y = fin['boxes'][:,7]
        fin.close()

        neg_idx = np.where(iou < neg_threshold)
        Y[neg_idx] = -1

        # select only 
        if load_for_train:
            sample_idx = np.where((iou <neg_threshold)|(iou >= pos_threshold))[0]
            X = X[sample_idx]
            Y = Y[sample_idx]

        return X, y_val_func(Y)
