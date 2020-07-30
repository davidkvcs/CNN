"""
Preprocessing script for dicom files for the example dataset. The script has been made specifically for the dataset, and will
not work for other datasets, unless changed to fit the new data.
"""

# Import needed python libraries:
import os
import os.path
import subprocess
import numpy as np
import pyminc.volumes.factory as pyminc
import argparse
#from CAAI import dcm
from rhscripts.conversion import rtx_to_mnc
from rhscripts.conversion import mnc_to_numpy
from rhscripts.kbq2suv_dgk import kbq2suv
from rhscripts import dcm

# Define function
def preprocess_dicom(input_path, clobber = True):
    # Step 1: Convert dicom files to minc format, using the minc toolbox. All series present in the input folder will be converted.
    #check if patient is already preprocessed
    #TODO: Load RT_STRUCT and CREATE .MNC ++ add new files and changes to GITHUB
    source_path = os.path.join(input_path,'dicom')#Define source 
    destination_path = os.path.join(input_path,'minc')#Define destination
    if ((os.path.exists(os.path.join(input_path,'dicom'))) & (clobber == True)):
        #Running this if folder "MINC" and "DICOM" already exist 
        print("MINC files already craeted. Remaking these, as clobber = True")
        
        #convert dicom images to .mnc files
        subprocess.call('dcm2mnc -usecoordinates -fname "%N_%A%m" -dname "" ' + source_path + "/* " + destination_path, shell=True) #-clobber

        #rename mnc files based on modality
        rename(input_path)

        #get dicom file paths
        rtstruct_ct_pt_path = get_rtstruct_ct_pet_filepath(source_path)

        #convert pet files from kbq to suv
        print('petfile_dicom = ' + rtstruct_ct_pt_path[2])
        print('petfile_mnc = ' + os.path.join(destination_path,'PT.mnc'))
        kbq2suv(rtstruct_ct_pt_path[2], os.path.join(destination_path,'PT.mnc'))

        #convert dicom structure files to .mnc
        rtstruct_container = get_rtx_dicom_container(destination_path, rtstruct_ct_pt_path[0])
        rtx_mnc_out_filepath = rtx_to_mnc(rtstruct_ct_pt_path[0], rtstruct_container, destination_path)


        #resample PET file to the CT file - projectspecific
        pt_mnc_path = rsl_pet_to_ct(destination_path)

        #resample RTX file to the CT file - 
        rtx_mnc_path = rsl_rtx_to_ct(destination_path, rtx_mnc_out_filepath)

        #create and save ct .npy file (memmap)
        ct_mnc_path = os.path.join(destination_path,'CT.mnc')
        save_dat_memmap(ct_mnc_path,pt_mnc_path,rtx_mnc_path)        
    

    elif ((os.path.exists(os.path.join(input_path,'dicom'))) & (clobber == False)):
        #Not running again if clobber = false
        print("MINC files already craeted. Not preprocessing again, as clobber = False")
    else:
        #First time you preprocess this is running 
        print( "Creating MINC-files for patient.")
        subprocess.call('dcm2mnc -usecoordinates -fname "%N_%A%m" -dname "" ' + input_path + "/* " + input_path, shell=True)
        reorganize(input_path)
        rename(input_path)

        #get dicom file paths
        rtstruct_ct_pt_path = get_rtstruct_ct_pet_filepath(source_path)

        #convert to suv
        print('petfile_dicom = ' + rtstruct_ct_pt_path[2])
        print('petfile_mnc = ' + os.path.join(destination_path,'PT.mnc'))
        kbq2suv(rtstruct_ct_pt_path[2], os.path.join(destination_path,'PT.mnc'))

        #convert dicom structure files to .mnc
        rtstruct_container = get_rtx_dicom_container(destination_path, rtstruct_ct_pt_path[0])
        rtx_mnc_out_filepath = rtx_to_mnc(rtstruct_ct_pt_path[0], rtstruct_container, destination_path)

        #resample PET file to the CT file - projectspecific
        pt_mnc_path = rsl_pet_to_ct(destination_path)

        #resample RTX file to the CT file - 
        rtx_mnc_path = rsl_rtx_to_ct(destination_path, rtx_mnc_out_filepath)

        #create and save ct .npy file (memmap)
        ct_mnc_path = os.path.join(destination_path,'CT.mnc')
        save_dat_memmap(ct_mnc_path,pt_mnc_path,rtx_mnc_path)        


def save_dat_memmap(ct_mnc_path,pt_mnc_path,rtx_mnc_path):
    #channel 1 holds CT, channel 2 holds pet, channel 3 holds structure
    ct = mnc_to_numpy(ct_mnc_path, datatype = 'float32')
    pt = mnc_to_numpy(pt_mnc_path, datatype = 'float32')
    struct = mnc_to_numpy(rtx_mnc_path, datatype = 'bool')

    dat = np.empty((ct.shape[0],ct.shape[1],ct.shape[2],3))
    dat[...,0] = ct
    dat[...,1] = pt
    dat[...,2] = struct
    print(dat.shape)

    path = os.path.dirname(ct_mnc_path)
    memmap_dat = np.memmap(path+'/dat.npy', dtype='float32', mode='w+', shape=dat.shape)
    memmap_dat[:] = dat[:]
    del memmap_dat

def rsl_rtx_to_ct(destination_path, rtx_mnc_out_filepath):
    #resample pet file like ct file
    #and save an npy file with same name
    
    rtx_mnc_like_ct = os.path.join(destination_path, os.path.splitext(os.path.basename(rtx_mnc_out_filepath))[0]+'_rsl.mnc')
    ct_mnc_file = os.path.join(destination_path,'CT.mnc')
    subprocess.call('mincresample -clobber -nearest_neighbour ' + rtx_mnc_out_filepath + ' ' + rtx_mnc_like_ct + ' -like ' + ct_mnc_file, shell=True) #-clobber      
    #delete the old pet file
    os.remove(rtx_mnc_out_filepath)

    return rtx_mnc_like_ct

    #open generated mnc file and save as numpy memmep
    #rtx_mnc_like_ct_np = mnc_to_numpy(rtx_mnc_like_ct)
    #rtx_npy_like_ct = os.path.join(destination_path, os.path.splitext(os.path.basename(rtx_mnc_out_filepath))[0]+'_rsl.npy')
    #rtx_mnc_like_ct_np_memmap = np.memmap(rtx_npy_like_ct, dtype = 'bool', mode = 'w+', shape = rtx_mnc_like_ct_np.shape)
    #rtx_mnc_like_ct_np_memmap[:] = rtx_mnc_like_ct_np[:]

def rsl_pet_to_ct(destination_path):
    #resample pet file like ct file
    #and save an npy file with same name

    pt_mnc_file = os.path.join(destination_path,'PT.mnc')
    pt_mnc_suv_file = os.path.join(destination_path,'PTsuv.mnc')
    pt_mnc_like_ct = os.path.join(destination_path,'PTsuv_rsl.mnc')
    ct_mnc_file = os.path.join(destination_path,'CT.mnc')
    subprocess.call('mincresample -clobber ' + pt_mnc_suv_file + ' ' + pt_mnc_like_ct + ' -like ' + ct_mnc_file, shell=True) #-clobber      
    #delete the old pet file
    os.remove(pt_mnc_file)
    os.remove(pt_mnc_suv_file)

    return pt_mnc_like_ct
 
    #open generated mnc file and save as numpy memmep
    #pt_mnc_like_ct_np = mnc_to_numpy(pt_mnc_like_ct)
    #pt_npy_like_ct = os.path.join(destination_path, os.path.splitext(os.path.basename(pt_mnc_suv_file))[0]+'_rsl.npy')
    #pt_mnc_like_ct_np_memmap = np.memmap(pt_npy_like_ct, dtype = 'float32', mode = 'w+', shape = pt_mnc_like_ct_np.shape)
    #pt_mnc_like_ct_np_memmap[:] = pt_mnc_like_ct_np[:]


def get_rtx_dicom_container(destination_path, rtstruct_path):
    #get PET series, CT series and "RTstruct reference series" UID
    rtstruct_ref_series_uid = dcm.get_rtx_referenced_series_instance_uid(rtstruct_path)
    ct_path = os.path.join(destination_path,'CT.mnc')
    pt_path = os.path.join(destination_path,'PT.mnc')
    ct_series_uid = get_dicom_image_uid(ct_path)
    pt_series_uid = get_dicom_image_uid(pt_path)
    if ct_series_uid == rtstruct_ref_series_uid:
        rtstruct_container = os.path.join(destination_path,'CT.mnc')
    elif pt_series_uid == rtstruct_ref_series_uid:
        rtstruct_container = os.path.join(destination_path,'PT.mnc')
    else:
        print('No matching series UID was found to RTSTRUCT')
        rtstruct_container = os.path.join(destination_path,'PT.mnc')    
    return rtstruct_container

def get_dicom_image_uid(input_path):
    proc = subprocess.Popen('mincinfo '+input_path+' -attvalue dicom_0x0020:el_0x000e',shell=True,stdout=subprocess.PIPE)
    output = proc.stdout.read()
    image_series_uid = output.decode("utf-8")
    image_series_uid = image_series_uid.rstrip()
    return image_series_uid

#rename dicom files according to modality
def get_rtstruct_ct_pet_filepath(input_path):
    '''
    Get modcality information from all files in header and return table with info
    Inputs: Path to a dicom directory
    
    Outputs:
        - file holding path to rtstruct
        - file holding path to one ct dicom file
        - file holding path to one pet dicom file
    
    TODO: Currently works if there is maximum of 1 RTSTRUCT file.

    '''
    extensions = [".ima", ".IMA", ".dcm", ".DCM"]
    for root, dirs, files in os.walk(input_path):
        #check if rtstruct_file.txt already exists
        output_rtx_filename = os.path.join(root,'rtx_file.txt')
        output_ct_filename = os.path.join(root,'ct_file.txt')
        output_pt_filename = os.path.join(root,'pt_file.txt')
        if os.path.isfile(output_rtx_filename) == True:
            with open(output_rtx_filename, 'r') as file:
                rtx_filepath = file.read().replace('\n', '')
                print('rtx_file.txt already exists and was loaded.')
        if os.path.isfile(output_ct_filename) == True:
            with open(output_ct_filename, 'r') as file:
                ct_filepath = file.read().replace('\n', '')
                print('ct_file.txt already exists and was loaded.')
        if os.path.isfile(output_pt_filename) == True:
            with open(output_pt_filename, 'r') as file:
                pt_filepath = file.read().replace('\n', '')
                print('pt_file.txt already exists and was loaded.')
        else:
            for file in files:
                filepath = os.path.join(root, file)
                for ext in extensions:
                    if file.endswith(ext):
                        modality = dcm.get_modality(filepath)
                        if modality == "RTSTRUCT":
                            #Save path to file, so this only runs once.
                            output_file = open(output_rtx_filename, 'w')
                            output_file.write(filepath)
                            output_file.close()
                            #print('Saved path to rtstruct in rtstruct_file.txt')
                            rtx_filepath = filepath
                        elif modality == "CT":
                            #Save path to file, so this only runs once.
                            output_file = open(output_ct_filename, 'w')
                            output_file.write(filepath)
                            output_file.close()
                            #print('Saved path to rtstruct in rtstruct_file.txt')
                            ct_filepath = filepath
                        elif modality == "PT":
                            #Save path to file, so this only runs once.
                            output_file = open(output_pt_filename, 'w')
                            output_file.write(filepath)
                            output_file.close()
                            #print('Saved path to rtstruct in rtstruct_file.txt')
                            pt_filepath = filepath
        return rtx_filepath, ct_filepath, pt_filepath

def rename(input_path):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mnc"):
                # Extract pt name from minc header (patient# in this case):
                #proc = subprocess.Popen('mincinfo '+(os.path.join(root, file))+' -attvalue dicom_0x0010:el_0x0010',shell=True,stdout=subprocess.PIPE)
                #output = proc.stdout.read()
                #pt = output.decode("utf-8")
                #pt = pt.rstrip()
                
                # Extract modality from minc header:
                proc = subprocess.Popen('mincinfo '+(os.path.join(root, file))+' -attvalue dicom_0x0008:el_0x0060',shell=True,stdout=subprocess.PIPE)
                output = proc.stdout.read()
                modality = output.decode("utf-8")
                modality = modality.rstrip()
                
                # Extract unit type from minc header:
                proc = subprocess.Popen('mincinfo '+(os.path.join(root, file))+' -attvalue dicom_0x0028:el_0x1054',shell=True,stdout=subprocess.PIPE)
                output = proc.stdout.read()
                unit = output.decode("utf-8")
                unit = unit.rstrip()

                #print('patient = ' + pt)
                #print('modality = ' + modality)
                #print('unit = ' + unit)
                
                # Create new name depending on attributes:
                new_name = modality + '.mnc'
                
                # Rename minc file:
                #print('will rund this command: mv '+(os.path.join(root, file))+' '+(os.path.join(root, new_name)))
                if not (os.path.join(root, file)) == (os.path.join(root, new_name)):
                    subprocess.call('mv '+(os.path.join(root, file))+' '+(os.path.join(root, new_name)),shell=True)

def reorganize(input_path):
    # Step 6: Organize files:
    # Make folders:
    subprocess.call('mkdir ' + input_path+'/dicom',shell=True)
    subprocess.call('mkdir ' + input_path+'/minc',shell=True)
    
    # Move files
    subprocess.call("mv " + os.path.join(input_path,"*.dcm ") + os.path.join(input_path,"dicom"), shell=True)
    subprocess.call("mv " + os.path.join(input_path,"*.ima ") + os.path.join(input_path,"dicom"), shell=True)
    subprocess.call("mv " + os.path.join(input_path,"*.DCM ") + os.path.join(input_path,"dicom"), shell=True)
    subprocess.call("mv " + os.path.join(input_path,"*.mnc ") + os.path.join(input_path,"minc"), shell=True)
   # subprocess.call("mv " + os.path.join(input_path,"*.npy ") + os.path.join(input_path,"minc"), shell=True)

'''
    # Step 5: Create numpy arrays.
    
    _ct_name = "CT_resampled.mnc"
    _highdose_name = "AVG_TrueX1-6_reshaped.mnc"
    _lowdose_name = "PET_TrueX1_reshaped.mnc"
    
    _dat_name = "dat_256_truex1_256_CT.npy"
    _res_name = "res_256_truex1_256_CT.npy"
    
    scale = 1 #scales the reference in the residual calculation: ref/scale. No need for scaling of breathhold data.

    # Load PET/CT input
    _lowdose = pyminc.volumeFromFile(os.path.join(input_path,pt+_lowdose_name)) # Load minc file
    _ct = pyminc.volumeFromFile(os.path.join(input_path,pt+_ct_name)) # Load minc file
    lowdose = np.array(_lowdose.data,dtype='double') # Create numpy array from minc data
    ct = np.array(_ct.data,dtype='double') # Create numpy array from minc data
    ct = np.swapaxes(np.swapaxes(ct,0,1),1,2) # Axis swapping
    lowdose = np.swapaxes(np.swapaxes(lowdose,0,1),1,2) # Axis swapping
    lowdose[np.where(lowdose<0)] = 0
    ct = ct
    ct[np.where(ct<-1024)] = -1024
    _lowdose.closeVolume()
    _ct.closeVolume()
    
    # Load reference averaged scan
    _highdose = pyminc.volumeFromFile(os.path.join(input_path,pt+_highdose_name)) # Load minc file
    highdose = np.array(_highdose.data,dtype='double') # Create numpy array from minc data
    highdose = np.swapaxes(np.swapaxes(highdose,0,1),1,2) # Axis swapping
    _highdose.closeVolume()
    highdose[np.where(highdose<0)] = 0
    
    # Compute residual image
    residual = np.true_divide(highdose,scale) - lowdose # Compute target residual
    
    # Fill out data matrix
    dat = np.empty((lowdose.shape[0],lowdose.shape[1],lowdose.shape[2],2))
    dat[...,0] = lowdose
    dat[...,1] = ct
    
    # Save the PET/CT matrix and residual reference       
    memmap_dat = np.memmap(input_path+'/'+_dat_name, dtype='double', mode='w+', shape=dat.shape)
    memmap_dat[:] = dat[:]
            
    del memmap_dat
            
    memmap_res = np.memmap(input_path+'/'+_res_name, dtype='double', mode='w+', shape=(residual.shape))
    memmap_res[:] = residual[:]
            
    del memmap_res
    
    # Step 6: Organize files:
    
    # Make folders:
    subprocess.call('mkdir ' + input_path+'/dicom',shell=True)
    subprocess.call('mkdir ' + input_path+'/minc',shell=True)
    
    # Move files
    subprocess.call("mv " + os.path.join(input_path,"*.dcm ") + os.path.join(input_path,"dicom"), shell=True)
    subprocess.call("mv " + os.path.join(input_path,"*.mnc ") + os.path.join(input_path,"minc"), shell=True)
    subprocess.call("mv " + os.path.join(input_path,"*.npy ") + os.path.join(input_path,"minc"), shell=True)
'''    

# Get dicom directory. The script can be run from the terminal with an input argument, 
# such as: python3 preprocess_dicom.py --input /users/mathias/deeplearning_guide/patient_data_raw/patient2
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess dicom files for later denoising.')
    parser.add_argument("--input", help="Specify the input folder path", type=str) # Takes input argument
    args = parser.parse_args()
    
    # Call function using input path:
    preprocess_dicom(args.input)


