
from DataAugmentation3D import DataAugmentation3D

"""

##############
Data Generator
##############

Please update the dat and tgt filenames, as well as matrix size and how the stacks are extracted

"""

class DataGenerator():

	def __init__(self, batch_size, img_res, input_channels, output_channels, augmentation, augmentation_params, data_pickle, data_folder):
		self.batch_size = batch_size
		self.img_res = img_res
		self.input_channels = input_channels
		self.output_channels = output_channels
		self.augmentation = augmentation
		self.augmentation_params = augmentation_params

		self.summary = pickle.load( open(data_pickle, 'rb') )
		self.root = data_folder

		# HARCODED FILENAMES!
		self.dat_name = 'dat_01_suv_ctnorm_double.npy'
		self.tgt_name = 'res_01_suv_double.npy'

		self.n_batches = self.summary['train']/self.batch_size if 'train' in self.summary else self.summary['train_0']/self.batch_size

	def generate(self, train_or_test):
		while 1:
			X, y = self.__data_generation(train_or_test)
			yield X, y

	def __data_generation(self, train_or_test):
		X = np.empty( (self.batch_size,) + self.img_res + (self.input_channels,) )
		y = np.empty( (self.batch_size,) + self.img_res + (self.output_channels,) )

		for i in range(self.batch_size):

			dat,tgt = self.load(train_or_test,load_mode='memmap')

			if train_or_test.startswith('train') and self.augmentation:
				dat, tgt = DataAugmentation3D(**self.augmentation_params).random_transform_batch(dat,tgt) # Hvorfor ny klasse hver gang?

			X[i,...] = dat
			y[i,...] = tgt

		return X,y

	def load(self, mode, z=None, return_studyid=False, load_mode='npy'):

	    indices = np.random.randint(0, len(self.summary[mode]), 1)
	    stats = self.summary[mode][indices[0]]

	    # --- Load data and labels 
	    fname_dat = '%s/%s/%s' % (self.data_folder, stats, self.dat_name)
	    fname_tgt = '%s/%s/%s' % (self.data_folder, stats, self.tgt_name)

	    if load_mode == 'npy':
	    	dat = np.load(fname)
			tgt = np.load(fname)
		elif load_mode == 'memmap':
			dat = np.memmap(fname, dtype='double', mode='r')
	    	dat = dat.reshape(128,128,111,2)
			tgt = np.memmap(fname, dtype='double', mode='r')
	    	tgt = tgt.reshape(128,128,111)

	    # --- Determine slice
	    if z == None:
	        z = np.random.randint(8,111-8,1)[0]
	    
	    dat_stack = dat[:,:,z-8:z+8,:]
	    tgt_stack = tgt[:,:,z-8:z+8]

	    if return_studyid:
	        return dat_stack, tgt_stack, stats
	    else:
	        return dat_stack, tgt_stack