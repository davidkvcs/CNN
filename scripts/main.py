from CAAI.train import CNN
from CAAI.predict import CNN as TrainedModel
import pickle, os
import numpy as np
from data_generator import DataGenerator
import pyminc.volumes.factory as pyminc

#Project specific parameters - must be set for each project
data_base_path = '/Volumes/my_passport/project_data/lymphoma-auto-contouring/patient_data_preprocessed/'
pickle_file = 'data_2fold.pickle'
x = 256 #image size in x-direction
y = 256 #image size in y-direction
z = 8 #number of input slices to network
d = 2 #number of input channels
n_slices = 111 #total slices in series
binary_output = 1#set to zero to get binary output by thresholding (for mask prediction)

#Must be set for each model trained
n_k_fold = 0 #change for each k-fold training in same version
version_num = 1 #change for each new version where new hyperparameters are used

def train_v1():
    
    cnn = CNN(model_name='v'+str(version_num),
              input_patch_shape=(x,y,z),
              input_channels=d,
              output_channels=1,
              batch_size=2,
              epochs=2000,
              learning_rate=1e-4,
              checkpoint_save_rate=50,
              loss_functions=[['mean_absolute_error',1]],
              data_pickle= data_base_path+pickle_file,
              data_folder= data_base_path,
              data_pickle_kfold=n_k_fold,
              network_architecture = 'unet_8_slice', #more archs can be added by configuring networks.py and and "def build_network" in train.py
              n_base_filters = 64 #can for instance use 64 or 32 
              )
    
    # Attach generator
    cnn.data_loader = DataGenerator(cnn.config)  

    cnn.print_config()

    final_model_name = cnn.train()    
    
    return final_model_name
    
def predict(modelh5name, model_name=None):
    
    modelbasename = os.path.splitext(os.path.basename(modelh5name))[0]
    model = TrainedModel(modelh5name)
    
    # Overwrite name if set
    if model_name:
        modelbasename = model_name
    
    summary = pickle.load( open(data_base_path+pickle_file, 'rb') )
    for pt in summary['valid_'+str(n_k_fold)]:
        predict_patient(pt,model,modelbasename)

def predict_patient(pt,model,modelbasename):
    _predictor_fname = pt+"AVG_TrueX1-6_reshaped.mnc"#Container, hvor prædikterede data indsættes i #project specific
    fname_dat = os.path.join(data_base_path,pt,'minc/pet-ct.npy')
    dat = np.memmap(fname_dat, dtype='double', mode='r')
    dat = dat.reshape(x,y,-1,2)
    
    print("Predicting volume for %s" % pt)
    predicted = np.empty((n_slices,x,y))
    for z_index in range(int(z/2),n_slices-int(z/2)):
        predicted_stack = model.predict(dat[:,:,z_index-int(z/2):z_index+int(z/2),:].reshape(1,x,y,z,d))
        if z_index == int(z/2):
            for ind in range(int(z/2)):
                predicted[ind,:,:] = predicted_stack[0,:,:,ind].reshape(x,y)
        if z_index == n_slices-int(z/2)-1:
            for ind in range(int(z/2)):
                predicted[z_index+ind,:,:] = predicted_stack[0,:,:,int(z/2)+ind].reshape(x,y)
        predicted[z_index,:,:] = predicted_stack[0,:,:,int(z/2)].reshape(x,y) 
    predicted_full = predicted
    predicted_full += np.swapaxes(np.swapaxes(dat[:,:,:,0],2,1),1,0)
    
    # Create minc file of predicted data.
    out_vol = pyminc.volumeLikeFile(os.path.join(data_base_path,pt,'minc',_predictor_fname),os.path.join(data_base_path,pt,'predicted_'+modelbasename+'_'+_predictor_fname))
    
    #thresholding (use if you need binary output)
    if binary_output ==1:
      thres = 0.5
      predicted_full[predicted_full[...,0] > thres] = 1
      predicted_full[predicted_full[...,0] <= thres] = 0  

    #save the data
    out_vol.data = predicted_full
    out_vol.writeFile()
    out_vol.closeVolume()

if __name__ == '__main__':
    model_name = train_v1()
    predict(model_name)