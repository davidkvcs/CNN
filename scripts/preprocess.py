# Run preprocessing for all subfolders in the patient_data_only_dicom folder.

import glob
from preprocess_dicom import preprocess_dicom
from rtx2mnc import rtx2mnc

datapath = '/project_data/hnc-auto-contouring/patient_data_preprocessing/patient_data_only_dicom'

# Run preprocessing script on all patient folders:
for path in glob.glob(datapath+'/*'):
    print('Preprocessing: '+path)
    preprocess_dicom(path)
    

# Convert delineation file to minc format:
'''    
# Collect input paths.
RT_list = glob.glob('/users/mathias/deeplearning_guide/patient_data_only_dicom/*/dicom/*RT.dcm')
RT_list.sort()
MNC_list = glob.glob('/users/mathias/deeplearning_guide/patient_data_only_dicom/*/minc/*AVG_TrueX1-6_reshaped.mnc')
MNC_list.sort()
MNC_name = list(MNC_list) # Copy list into new list.

# Remove last part of string and replace it with 'RT.mnc'.
for i in range(len(MNC_name)):
    MNC_name[i] = MNC_name[i][:-25]+'RT.mnc'

# Use list information as input to rtx2mnc function.
for i in range(len(MNC_list)):
    print('Converting '+RT_list(i))
    rtx2mnc(RT_list[i],MNC_list[i],MNC_name[i])
    
    
'''