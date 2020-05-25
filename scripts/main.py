# Import python libraries:
from CAAI.train import CNN
from CAAI.predict import CNN as TrainedModel
import pickle, os
import numpy as np
from data_generator import DataGenerator
import pyminc.volumes.factory as pyminc

data_path = '/homes/kovacs/project_data/lymphoma-auto-contouring/'

# Define training function:
def train_v1():
    
    # Parameter definition for the network. These must be changed to fit your own data. Other parameters exist. See train.py
    cnn = CNN(model_name='v2_test_david',
              input_patch_shape=(256,256,8), # Dimensions of model input in the form (x,y,z)
              input_channels=2, # Number of model input channels. In this case, PET in 1st channel, CT in 2nd channel.
              output_channels=1, # Number of model output channels. In this case, a PET image as output.
              batch_size=1, # Number of inputs used in batch normalization. 1 = no batch normalization.
              epochs=100, # Number of training epochs. A stopping critera can also be implemented.
              learning_rate=1e-4, # Set learning rate.
              checkpoint_save_rate=50, # Saves model each # epoch.
              loss_functions=[['mean_absolute_error',1]], # Define loss function
              data_pickle= data_path+'patient_data_preprocessed/data_2fold.pickle', # K-fold split file
              data_folder= data_path+'patient_data_preprocessed',
              data_pickle_kfold=0 # K-fold. For 2-fold validation, each fold must be run subsequently. (First fold 0, then fold 1)
              )
    
    # Attach generator
    cnn.data_loader = DataGenerator(cnn.config) # Load data generator

    cnn.print_config() # Print network configurations

    final_model_name = cnn.train() # Create training network
    
    return final_model_name


# Define main prediction function:
def predict(modelh5name, model_name=None):
    
    modelbasename = os.path.splitext(os.path.basename(modelh5name))[0]
    model = TrainedModel(modelh5name)
    
    # Overwrite name if set
    if model_name:
        modelbasename = model_name
    
    # Load k-fold file.
    summary = pickle.load( open(data_path+'patient_data_preprocessed/data_2fold.pickle', 'rb') )
    for pt in summary['valid_0']: # K-fold. For 2-fold validation, each fold must be run subsequently. (First valid_0, then valid_1)
        predict_patient(pt,model,modelbasename)


# Define single patient prediction function:
def predict_patient(pt,model,modelbasename):
    _lowdose_name = "minc/"+pt+"PET_TrueX1_reshaped.mnc"
    data_folder = data_path+'patient_data_preprocessed'
    fname_dat = os.path.join(data_folder,pt,'minc/dat_256_truex1_256_CT.npy')
    dat = np.memmap(fname_dat, dtype='double', mode='r')
    dat = dat.reshape(256,256,-1,2)
    
    print("Predicting volume for %s" % pt)
    predicted = np.empty((111,256,256)) # Create empty matrix, which will be filled with predicted data.
    x,y,z,d = 256,256,8,2 # Define dimensions.

    for z_index in range(int(z/2),111-int(z/2)): # Loop over empty matrix.
        predicted_stack = model.predict(dat[:,:,z_index-int(z/2):z_index+int(z/2),:].reshape(1,x,y,z,d)) # Predict data slice.
        if z_index == int(z/2): # Handle edge case.
            for ind in range(int(z/2)):
                predicted[ind,:,:] = predicted_stack[0,:,:,ind].reshape(256,256) # Fill out matrix with prediction.
        if z_index == 111-int(z/2)-1: # Handle edge case.
            for ind in range(int(z/2)):
                predicted[z_index+ind,:,:] = predicted_stack[0,:,:,int(z/2)+ind].reshape(256,256) # Fill out matrix with prediction.
        predicted[z_index,:,:] = predicted_stack[0,:,:,int(z/2)].reshape(256,256) # Fill out matrix with prediction.
    predicted_full = predicted
    predicted_full += np.swapaxes(np.swapaxes(dat[:,:,:,0],2,1),1,0)
    
    # Create minc file of predicted data.
    out_vol = pyminc.volumeLikeFile(os.path.join(data_folder,pt,_lowdose_name),os.path.join(data_folder,pt,"minc",'predicted_'+modelbasename+'_'+_lowdose_name[5:]))
    out_vol.data = predicted_full
    out_vol.writeFile()
    out_vol.closeVolume()


# Start training of model and prediction of test data:
if __name__ == '__main__':
    model_name = train_v1()
    predict(model_name)
    #predict('v1_e10_bz1_lr1.0E-04_noDA_noTL_LOG0.h5')