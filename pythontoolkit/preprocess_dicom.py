"""
Preprocessing script for dicom files for the example dataset. The script has been made specifically for the dataset, and will
not work for other datasets, unless changed to fit the new data.
"""

# Import needed python libraries:
import os
import subprocess
import numpy as np
import pyminc.volumes.factory as pyminc
import argparse


# Define function
def preprocess_dicom(input_path, clobber = True):
    # Step 1: Convert dicom files to minc format, using the minc toolbox. All series present in the input folder will be converted.
    #check if patient is already preprocessed
    if ((os.path.exists(os.path.join(input_path,'dicom'))) & (clobber == True)):
        print("MINC files already craeted. Now remaking these, as clobber = True")
        source_path = os.path.join(input_path,'dicom')
        destination_path = os.path.join(input_path,'minc')
        subprocess.call('dcm2mnc -clobber -usecoordinates -fname "%N_%A%m" -dname "" ' + source_path + "/* " + destination_path, shell=True)
    elif ((os.path.exists(os.path.join(input_path,'dicom'))) & (clobber == False)):
        print("MINC files already craeted. Not preprocessing again, as clobber = False")
    else:
        print( "Creating MINC-files for patient.")
        subprocess.call('dcm2mnc -usecoordinates -fname "%N_%A%m" -dname "" ' + input_path + "/* " + input_path, shell=True)
    reorganize(input_path)

def rename(input_path):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mnc"):

                # Extract pt name from minc header (patient# in this case):
                proc = subprocess.Popen('mincinfo '+(os.path.join(root, file))+' -attvalue dicom_0x0010:el_0x0010',shell=True,stdout=subprocess.PIPE)
                output = proc.stdout.read()
                pt = output.decode("utf-8")
                pt = pt.rstrip()
                
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
                
                # Create new name depending on attributes:
                if modality == 'CT':
                    new_name = pt+modality+'.mnc'
                elif unit == 'BQML':
                    new_name = pt+'PET_TrueX1.mnc'
                else:
                    new_name = pt+'AVG_TrueX1-6.mnc'
                
                # Rename minc file:
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
    subprocess.call("mv " + os.path.join(input_path,"*.npy ") + os.path.join(input_path,"minc"), shell=True)


'''
    # Step 2: Rename minc files.
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mnc"):

                # Extract pt name from minc header (patient# in this case):
                proc = subprocess.Popen('mincinfo '+(os.path.join(root, file))+' -attvalue dicom_0x0010:el_0x0010',shell=True,stdout=subprocess.PIPE)
                output = proc.stdout.read()
                pt = output.decode("utf-8")
                pt = pt.rstrip()
                
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
                
                # Create new name depending on attributes:
                if modality == 'CT':
                    new_name = pt+modality+'.mnc'
                elif unit == 'BQML':
                    new_name = pt+'PET_TrueX1.mnc'
                else:
                    new_name = pt+'AVG_TrueX1-6.mnc'
                
                # Rename minc file:
                subprocess.call('mv '+(os.path.join(root, file))+' '+(os.path.join(root, new_name)),shell=True)
    
    
    # Step 3: Resizing of PET files from 400 to 256. Output file is given the extension _reshaped:
                
    # Loop through input folder to find .mnc PET files, that has not been reshaped already.
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mnc") and file.find("PET")!=-1 and file.find("reshaped")==-1:
                
                # Reshapes .mnc PET files using the MINC toolbox.
                subprocess.call("mincreshape -clobber "+os.path.join(root, file)+" "+os.path.join(root, file)[:-4]\
                                +"_reshaped.mnc"+" -start 0,72,72 -count 111,256,256",shell=True)
    
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mnc") and file.find("AVG")!=-1 and file.find("reshaped")==-1:
                
                # Reshapes .mnc PET files using the MINC toolbox.
                subprocess.call("mincreshape -clobber "+os.path.join(root, file)+" "+os.path.join(root, file)[:-4]\
                                +"_reshaped.mnc"+" -start 0,72,72 -count 111,256,256",shell=True)
    
    
    # Step 4: Register CT to PET image. Output file is given the extension _resampled:
       
    # Loop through input folder to find .mnc CT files, that has not been resampled already.
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".mnc") and file.find("CT")!=-1 and file.find("resampled")==-1:
                
                # Resamples .mnc CT files using the MINC toolbox.
                subprocess.call("mincresample "+os.path.join(root, file)+" "+os.path.join(root, file)[:-4]+\
                                "_resampled.mnc"+" -like "+os.path.join(root,pt + "PET_TrueX1_reshaped.mnc")+\
                                " -clobber -fillvalue -1024",shell=True)   
    
    
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


