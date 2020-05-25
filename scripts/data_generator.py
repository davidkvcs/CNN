# Import python libraries:
from CAAI.DataAugmentation3D import DataAugmentation3D
import pickle
import numpy as np


class DataGenerator():

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.img_res = config['input_patch_shape']
        self.input_channels = config['input_channels']
        self.output_channels = config['output_channels']
        self.augmentation = config['augmentation']
        self.augmentation_params = config['augmentation_params']
        self.data_augmentation = DataAugmentation3D(**self.augmentation_params)

        self.summary = pickle.load( open(config['data_pickle'], 'rb') )
        self.data_folder = config['data_folder']

        # HARCODED FILENAMES. Change to fit your own data.
        self.dat_name = 'minc/dat_256_truex1_256_CT.npy'
        self.tgt_name = 'minc/res_256_truex1_256_CT.npy'

        self.n_batches = len(self.summary['train']) if 'train' in self.summary else len(self.summary['train_0'])
        self.n_batches /= self.batch_size

    def generate(self, train_or_test):
        while 1:
            X, y = self.__data_generation(train_or_test)
            yield X, y

    def __data_generation(self, train_or_test):
        X = np.empty( (self.batch_size,) + self.img_res + (self.input_channels,) )
        y = np.empty( (self.batch_size,) + self.img_res + (self.output_channels,) )

        for i in range(self.batch_size):

            dat,tgt = self.load(train_or_test,load_mode='memmap')

            X[i,...] = dat
            y[i,...] = tgt.reshape(self.img_res + (self.output_channels,))
            
        if train_or_test.startswith('train') and self.augmentation:
            X, y = self.data_augmentation.random_transform_batch(X,y)

        return X,y

    def load(self, mode, z=None, return_studyid=False, load_mode='npy'):

        indices = np.random.randint(0, len(self.summary[mode]), 1)
        stats = self.summary[mode][indices[0]]

        # --- Load data and labels 
        fname_dat = '%s/%s/%s' % (self.data_folder, stats, self.dat_name)
        fname_tgt = '%s/%s/%s' % (self.data_folder, stats, self.tgt_name)

        if load_mode == 'npy':
            dat = np.load(fname_dat)
            tgt = np.load(fname_tgt)
        elif load_mode == 'memmap':
            dat = np.memmap(fname_dat, dtype='double', mode='r')
            dat = dat.reshape(256,256,-1,2)
            tgt = np.memmap(fname_tgt, dtype='double', mode='r')
            tgt = tgt.reshape(256,256,-1)

        # --- Determine slice
        if z == None:
            z = np.random.randint(4,111-4,1)[0]
        
        dat_stack = dat[:,:,z-4:z+4,:]
        tgt_stack = tgt[:,:,z-4:z+4]

        if return_studyid:
            return dat_stack, tgt_stack, stats
        else:
            return dat_stack, tgt_stack
        
        
        