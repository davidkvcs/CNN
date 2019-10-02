from train import CNN
from predict import CNN as TrainedModel
import pickle, os
import numpy as np

def train_v1():
    cnn = CNN(model_name='lowdose_cardiac_test_v2',
              input_patch_shape=(128,128,16),
              input_channels=2,
              output_channels=1,
              batch_size=2,
              epochs=50,
              learning_rate=1e-3,
              checkpoint_save_rate=10,
              loss_functions=[['mean_absolute_error',1]],
              data_pickle='test_dat.pickle',
              data_folder='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc'
              )

    cnn.print_config()

    cnn.train()
    
def predict_v1_fromh5():
    modelh5name = 'lowdose_cardiac_test_v1_e50_bz2_lr1.0E-04_DA_noTL.h5'
    modelbasename = os.path.splitext(os.path.basename(modelh5name))[0]
    model = TrainedModel(modelh5name)
    
    summary = pickle.load( open('test_dat.pickle', 'rb') )
    for pt in summary['valid']:
        predict_patient(pt,model,modelbasename)

def predict_v1_from_config():
    pass

def predict_patient(pt,model,modelbasename):
#    dat,res,pt_name = data.load_all_ctnorm_double('valid_%s' % LOG, ind=pt)
#    
    print("Predicting volume for %s" % pt)
#    predicted = np.empty((111,128,128))
#    
#    for z_index in range(int(z/2),111-int(z/2)):
#        predicted_stack = model.predict(dat[:,:,z_index-int(z/2):z_index+int(z/2),:].reshape(1,x,y,z,d))
#        if z_index == int(z/2):
#            for ind in range(int(z/2)):
#                predicted[ind,:,:] = predicted_stack[0,:,:,ind].reshape(128,128)
#        if z_index == 111-int(z/2)-1:
#            for ind in range(int(z/2)):
#                predicted[z_index+ind,:,:] = predicted_stack[0,:,:,int(z/2)+ind].reshape(128,128)
#        predicted[z_index,:,:] = predicted_stack[0,:,:,int(z/2)].reshape(128,128) 
#    predicted_full = predicted
#    if predict_noise_or_denoised == 'noise':
#        predicted_full += np.swapaxes(np.swapaxes(dat[:,:,:,0],2,1),1,0)
#    
#    model_name_ = model_name.split('/')[-1] if not model_name.endswith('.h5') else model_name.split('/')[-1][:-3]
#    out_vol = pyminc.volumeLikeFile(os.path.join('../PETrecon/HjerteFDG_mnc',pt_name,_lowdose_name),os.path.join('../PETrecon/HjerteFDG_mnc',pt_name,'predicted_'+model_name_+'_'+_lowdose_name))
#    out_vol.data = predicted_full
#    out_vol.writeFile()
#    out_vol.closeVolume()
    pass

if __name__ == '__main__':
    model_name = train_v1()
    #predict_v1_fromh5()
    #predict_v1_fromconfig()
    
    #predict_v1_fromh5()
    
    
