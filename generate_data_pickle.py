# -*- coding: utf-8 -*-

import pickle
import os

"""

Save pickle file of train and validation pts
Should have indexes "train" and "test" or "train_X" and "valid_X" 
where X is integer from 0, representing the LOG in k-fold.

"""

summary = { 'train': [], 'valid': [] }

pts = os.listdir('/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc')

summary['train'].append(pts[0])
summary['train'].append(pts[1])
summary['valid'].append(pts[10])
summary['valid'].append(pts[11])

with open('test_dat.pickle', 'wb') as file_pi:
    pickle.dump(summary,file_pi)