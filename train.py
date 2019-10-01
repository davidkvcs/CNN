
import os
import pickle
from glob import glob
import networks
import json
from data_generator import DataGenerator
from decimal import Decimal

"""

TODO:
 - Check content of existing CONFIG matches new run if continue
 - Delete checkpoints if running overwriting already trained model.
 - Generate data pickle file


"""

class CNN():
	def __init__(self,**kwargs):
		self.config = dict()
		self.config["model_name"] = 'PROJECT_NAME_WITH_VERSION_NUMBER'
		self.config["overwrite"] = False
		self.config["input_patch_shape"] = (16,192,240)
		self.config["input_channels"] = 2
		self.config["output_channels"] = 1
		self.config["batch_size"] = 2
		self.config["epochs"] = 100
		self.config["checkpoint_save_rate"] = 10
		self.config["intial_epoch"] = 0
		self.config["learning_rate"] = 1e-4
		self.config["loss_functions"] = [ ['mse',1] ]
		self.config["data_folder"] = '' # Path to folder containing data
		self.config["data_pickle"] = '' # Path to pickle containing train/validation splits
		self.config["data_pickle_kfold"] = None # Set to fold if k-fold training is applied (key will e.g. be train_0 and valid_0)
		self.config["train_pts"] = 'train' if self.config['data_pickle_kfold'] is None else 'train_{}'.format(self.config['data_pickle_kfold'])
		self.config["valid_pts"] = 'valid' if self.config['data_pickle_kfold'] is None else 'valid_{}'.format(self.config['data_pickle_kfold'])
		self.config["pretrained_model"] = None # If transfer learning from other model (not used if resuming training, but keep for model_name's sake)
		self.config["augmentation"] = True
		self.config["augmentation_params"] = {
													#'rotation_range': [5,5,5],
													'shift_range': [0.05,0.05,0.05],
													'shear_range': [2,2,0],
													'zoom_lower' : [0.9,0.9,0.9],
													'zoom_upper' : [1.2,1.2,1.2],
													'zoom_independent' : True,
													'flip_axis' : [1, 2],          
													#'X_shift_voxel' :[2, 2, 0],
													#'X_add_noise' : 0.1,
													'fill_mode' : 'reflect'
		}

		# Config specific for network architecture
		self.config['n_base_filters'] = 32

		# Update with overwritten params
		for k,v in kwargs.iteritems():
			if k in self.config:
				self.config[k] = v

		# Generate full model name
		self.config["model_name"] = self.generate_model_name_from_params()

		# Check if model has been trained (and can be overwritten), or if we should resume from checkpoint
		self.check_model_existance()

		""" SETUP MODEL """
		# Setup generators
		self.data_loader = DataGenerator(batch_size=self.config['batch_size'], 
										 img_res=self.config['input_patch_shape'], 
										 input_channels=self.config['input_channels'], 
										 output_channels=self.config['output_channels'],
										 augmentation=self.config['augmentation'],
										 augmentation_params=self.config['augmentation_params'],
										 data_pickle=self.config['data_pickle'],
										 data_folder=self.config['data_folder']
										)
		self.training_generator = self.data_loader.generate( self.config['train_pts'] )
		self.validation_generator = self.data_loader.generate( self.config['valid_pts'] )

		# Setup network
		self.network = build_unet()
		optimizer = Adam(self.config['learning_rate'])
		loss, loss_weights = [ l[0], l[1] for l in self.config['loss_functions'] ]
		self.network.compile(loss = loss, loss_weights = loss_weights, optimizer = optimizer)

		# Setup callbacks
		self.callbacks_list = setup_callbacks()

	def setup_callbacks(self):

		# Checkpoints
		os.makedirs('checkpoint', exists_ok=True)
		checkpoint_file='checkpoint/{}/{}_e{epoch:02d}_{val_loss:.2f}.h5'.format(self.config["model_name"],self.config["model_name"])
		checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False, mode='min',period=self.config["checkpoint_save_rate"])

		# Tensorboard		
		os.makedirs('logs', exists_ok=True)
		TB_file=os.path.join('logs',self.config["model_name"])
		TB = TensorBoard(log_dir = TB_file)

		return [checkpoint, TB]

	def build_unet(self):
		inputs = Input(shape=(self.config['input_patch_shape'],)+(self.config['input_channels'],))
		outputs = unet(inputs,f=self.config['n_base_filters'],dims_out=self.config['output_channels'])
		return Model(inputs=inputs,outputs=outputs)

	def generate_model_name_from_params(self):
		# Build full model name
		model_name = self.config['model_name']

		model_name += '_e{}'.format( self.config['epochs'] )
		model_name += '_bz{}'.format( self.config['batch_size'] )
		model_name += '_lr{:.1E}'.format( self.config['learning_rate'] )
		model_name += '_DA' if self.config['augmentation']  else '_noDA'
		model_name += '_TL' if self.config['pretrained_model'] is not None else '_noTL'
		model_name += '_LOG%d' % self.config['data_pickle_kfold'] if self.config['data_pickle_kfold'] is not None else ''

		return model_name

	def get_initial_epoch_from_file(f):
		last_epoch = f.split('_')[-2]
		assert last_epoch.startswith('e') # check that it is indeed the epoch part of the name that we extract
		return int(last_epoch[1:]) # extract only integer part of eXXX

	def check_model_existance(self):
		# Check if config file already exists
		if os.path.exists( '{}.pkl'.format( self.config['model_name'] ) ):
			
			# Check if model has been trained completely:
			if os.path.exists( '{}.h5'.format( self.config['model_name'] ) ) and not self.config['overwrite']:
				print("The model {} has already been trained, and you specified not to overwrite it. Stopping training.".format( self.config['model_name'] ))
				exit(-1)

			# The model has not finished training
			elif not os.path.exists( '{}.h5'.format( self.config['model_name'] ) ):

				# Check that config is the same..
				# TODO
				
				# Get checkpoint models if exists
				checkpoint_models = glob('checkpoint/{}_e*.h5'.format( self.config['model_name'] ))
				
				# Set resume model from latest checkpoint
				if len(checkpoint_models) > 0:
					self.config['model_resume'] = checkpoint_models[-1]
					self.config['initial_epoch'] = self.get_initial_epoch_from_file(checkpoint_models[-1])

			# else -> model exists but we specified to overwrite, so we do so, without loading from the checkpoint folder. 
			# OBS: The checkpoints should probably be cleared before starting?

	def print_config(self):
		print(json.dumps(self.config, indent = 4))

	def set(self,key,value):
		self.config[key] = value

	def train(self):

		# Save current configs to file.
		os.makedirs('configs', exists_ok=True)
		with open('configs/{}.pkl'.format(self.config["model_name"]), 'wb') as file_pi:
			pickle.dump(self.config, file_pi)

		history = model.fit_generator(	generator = self.training_generator,
										steps_per_epoch = self.data_loader.n_batches,
										validation_data = self.validation_generator,
										validation_steps = 1,
										epochs = self.config['epochs'],
										verbose = 1,
										callbacks = self.callbacks_list,
										initial_epoch = self.config['initial_epoch'] )

		# Save model
		model.save('{}.h5'.format( self.config['model_name'] ))

		# Save history file
		with open('{}_history.pickle', 'wb') as history_pi:
			pickle.dump( history.history, history_pi )
