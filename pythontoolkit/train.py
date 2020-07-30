# Import python libraries:
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from glob import glob
from CAAI import networks
import json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.optimizers import Adam


# Define Convolutional Neural Network class.
class CNN():
    
    # Define default configurations, which will be used if no other configurations are defined.
    def __init__(self,**kwargs):
        self.config = dict()
        self.config["model_name"] = 'PROJECT_NAME_WITH_VERSION_NUMBER'
        self.config["overwrite"] = False
        self.config["input_patch_shape"] = (8,256,256)
        self.config["input_channels"] = 2
        self.config["output_channels"] = 1
        self.config["batch_size"] = 1
        self.config["epochs"] = 1000
        self.config["checkpoint_save_rate"] = 10
        self.config["initial_epoch"] = 0
        self.config["learning_rate"] = 1e-4
        self.config["data_folder"] = '' # Path to folder containing data
        self.config["data_pickle"] = '' # Path to pickle containing train/validation splits
        self.config["data_pickle_kfold"] = None # Set to fold if k-fold training is applied (key will e.g. be train_0 and valid_0)
        self.config["pretrained_model"] = None # If transfer learning from other model (not used if resuming training, but keep for model_name's sake)
        self.config["augmentation"] = False
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

        self.config['model_resume'] = None

        # Config specific for network architecture
        self.config["network_architecture"] = 'unet'
        self.config['n_base_filters'] = 64
        self.custom_network_architecture = None
        
        # Metrics and loss functions
        self.loss_functions = [ ['mse',1] ]
        self.metrics = ['accuracy']
        self.custom_objects = dict()
        self.is_compiled = False

        # Update with overwritten params
        for k,v in kwargs.items():
            if k in self.config:
                self.config[k] = v
                
        # Update train and valid flags after pickle has been set
        self.config["train_pts"] = 'train' if self.config['data_pickle_kfold'] is None else 'train_{}'.format(self.config['data_pickle_kfold'])
        self.config["valid_pts"] = 'valid' if self.config['data_pickle_kfold'] is None else 'valid_{}'.format(self.config['data_pickle_kfold'])

        # Generate full model name
        self.config["model_name"] = self.generate_model_name_from_params()

        # Check if model has been trained (and can be overwritten), or if we should resume from checkpoint
        self.check_model_existance()


    def setup_callbacks(self):

        # Checkpoints
        os.makedirs('checkpoint/{}'.format(self.config['model_name']), exist_ok=True)
        checkpoint_file=os.path.join('checkpoint',self.config["model_name"],'e{epoch:02d}.h5')
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=1, save_best_only=False, mode='min',save_freq=int(self.config['checkpoint_save_rate']*self.data_loader.n_batches))

        # Tensorboard        
        os.makedirs('logs', exist_ok=True)
        TB_file=os.path.join('logs',self.config["model_name"])
        TB = TensorBoard(log_dir = TB_file)

        return [checkpoint, TB]

    
    def compile_network(self):
        
        """ SETUP MODEL """
        
        # Setup network
        
        # Resume previous training:
        if self.config['initial_epoch'] > 0:
            print("Resuming from model: {}".format(self.config['model_resume']))
            self.model = load_model(self.config['model_resume'], custom_objects=self.custom_objects)
        # Transfer learn from model saved new-style
        elif self.config['pretrained_model'] and self.config['pretrained_models'].endswith('.h5'):    
            print("Transfer learning from model: {}".format(self.config['pretrained_model']))
            self.model = load_model(self.config['model_resume'], custom_objects=self.custom_objects)
        else:
            # TL form model saved old-style
            if self.config['pretrained_model']:
                print("Transfer learning from model: {}".format(self.config['pretrained_model']))
                self.model = self.load_model_w_json(self.config['pretrained_model'])
            # No TL, build from scratch
            else:
                self.model = self.build_network()
        
            optimizer = Adam(self.config['learning_rate'])
            loss, loss_weights = zip(*self.loss_functions)
            loss = list(loss)
            loss_weights = list(loss_weights)
            
            self.model.compile(loss = loss, loss_weights = loss_weights, optimizer = optimizer, metrics=self.metrics)
            
        self.is_compiled = True

    
    def load_model_w_json(self,model):
        modelh5name = os.path.join( os.path.dirname(model), os.path.splitext(os.path.basename(model))[0]+'.h5' )
        json_file = open(model,'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(modelh5name)
        return model


    def build_network(self,inputs=None):
        if not inputs:
            inputs = Input(shape=self.config['input_patch_shape']+(self.config['input_channels'],))
            
        if self.config['network_architecture'] == 'unet':
            outputs = networks.unet(inputs,f=self.config['n_base_filters'],dims_out=self.config['output_channels'])
            
        elif self.config['network_architecture'] == 'custom' and not self.custom_network_architecture == None:
            outputs = self.custom_network_architecture(inputs,config=self.config)
        
        else:
            print("You are using a network that I dont know..")
            exit(-1)

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


    def get_initial_epoch_from_file(self,f):
        last_epoch = f.split('/')[-1].split('_')[0]
        assert last_epoch.startswith('e') # check that it is indeed the epoch part of the name that we extract
        return int(last_epoch[1:-3]) # extract only integer part of eXXX


    def check_model_existance(self):
        # Check if config file already exists
        if os.path.exists( 'configs/{}.pkl'.format( self.config['model_name'] ) ):
            
            # Check if model has been trained completely:
            if os.path.exists( '{}.h5'.format( self.config['model_name'] ) ) and not self.config['overwrite']:
                print("The model {} has already been trained, and you specified not to overwrite it. Stopping training.".format( self.config['model_name'] ))
                exit(-1)

            # The model has not finished training
            elif not os.path.exists( '{}.h5'.format( self.config['model_name'] ) ):

                # Check that config is the same..
                # TODO
                
                # Get checkpoint models if exists
                checkpoint_models = glob('checkpoint/{}/e*.h5'.format( self.config['model_name'] ))
                # Set resume model from latest checkpoint
                if len(checkpoint_models) > 0:
                    self.config['model_resume'] = checkpoint_models[-1]
                    self.config['initial_epoch'] = int(self.get_initial_epoch_from_file(checkpoint_models[-1]))
            # else -> model exists but we specified to overwrite, so we do so, without loading from the checkpoint folder. 
            # OBS: The checkpoints should probably be cleared before starting?


    def print_config(self):
        print(json.dumps(self.config, indent = 4))


    def set(self,key,value):
        self.config[key] = value

    def plot_model(self):
        # Compile network if it has not been done:
        if not self.is_compiled:
            self.compile_network()

        #tf.keras.utils.plot_model(self.model, 
         #   show_shapes=True, 
          #  to_file='model_fig.png')

    def train(self):

        # Setup callbacks
        self.callbacks_list = self.setup_callbacks()
        
        # Compile network if it has not been done:
        if not self.is_compiled:
            self.compile_network()
        
        print(self.model.summary())

        # Check if data generators has been attached
        if hasattr(self,'data_loader'):
            #self.training_generator = self.data_loader.generate( self.config['train_pts'] )
            #self.validation_generator = self.data_loader.generate( self.config['valid_pts'] )

            # Updated to TFv2 generator
            generator_shape_input = self.config["input_patch_shape"]+tuple([self.config["input_channels"]])
            generator_shape_output = self.config["input_patch_shape"]+tuple([self.config["output_channels"]])
            self.training_generator = tf.data.Dataset.from_generator(
                lambda: self.data_loader.generate( self.config['train_pts'] ), 
                output_types=(tf.float32,tf.float32), 
                output_shapes=(tf.TensorShape(generator_shape_input),tf.TensorShape(generator_shape_output)))
            self.validation_generator = tf.data.Dataset.from_generator(
                lambda: self.data_loader.generate( self.config['valid_pts'] ), 
                output_types=(tf.float32,tf.float32), 
                output_shapes=(tf.TensorShape(generator_shape_input),tf.TensorShape(generator_shape_output)))

            self.training_generator = self.training_generator.batch(self.config["batch_size"])
            self.validation_generator = self.validation_generator.batch(self.config["batch_size"])
   
        else:
            print("No data generator was attached.")
            exit(-1)

        # Save current configs to file.
        os.makedirs('configs', exist_ok=True)
        with open('configs/{}.pkl'.format(self.config["model_name"]), 'wb') as file_pi:
            pickle.dump(self.config, file_pi)

        history = self.model.fit(  self.training_generator,
                                   steps_per_epoch = self.data_loader.n_batches,
                                   validation_data = self.validation_generator,
                                   validation_steps = 1,
                                   epochs = self.config['epochs'],
                                   verbose = 1,
                                   callbacks = self.callbacks_list,
                                   initial_epoch = self.config['initial_epoch'] )

        # Save model
        self.model.save('{}.h5'.format( self.config['model_name'] ))

        # Save history file
        with open('{}_history.pickle'.format(self.config['model_name']), 'wb') as history_pi:
            pickle.dump( history.history, history_pi )
        
        return self.config['model_name']+'.h5'