import tensorflow
import numpy as np
import pickle

out_pickle_fn = '/home/mohammad/Projects/deepfnftf2/logs-grid/deepfnf-orig/train/params/latest_parameters.pickle'
dummy_pickle_fn = '/home/mohammad/Projects/deepfnftf2/logs-grid/deepfnf-05/train/params/latest_parameters.pickle'
npz_fn = '/home/mohammad/Projects/deepfnftf2/deepfnf_orig_model.npz'
model_params = dict(np.load(npz_fn))


with open(dummy_pickle_fn,'rb') as fd:
    dummy_pickle = pickle.load(fd)

model_pickle = {'step':dummy_pickle['step'],
                'params':dict(model_params),
                'idx':dummy_pickle['idx'],
                'state':dummy_pickle['state']
                }

with open(out_pickle_fn,'wb') as fd:
    obj = pickle.dump(model_pickle, fd)