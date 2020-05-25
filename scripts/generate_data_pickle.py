# Import python libraries:
from sklearn.model_selection import KFold
import numpy as np
import os
import pickle


# Create 2-fold split of the example dataset:
datafolder='/users/mathias/deeplearning_guide/patient_data_only_dicom'

patients = [f for f in os.listdir(datafolder)]
patients = np.array(patients)

# Define splits
kf = KFold(n_splits=2,shuffle=True)
kf.get_n_splits(patients)

# Fill out split dictionary
data = {}
for G, (train,test) in enumerate(kf.split(patients)):
    print(G,len(train),len(test))
    data['train_%d' % G] = patients[train]
    data['valid_%d' % G] = patients[test]

# Save k-fold split file  
pickle.dump( data, open('/users/mathias/deeplearning_guide/patient_data_only_dicom/data_2fold.pickle','wb') )


