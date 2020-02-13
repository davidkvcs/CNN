# CAAI CNN template
Scripts used at CAAI group at Rigshospitalet

## HOW TO USE THE SCRIPTS
Once the python libraries for training and predicting a CNN is installed, 
you can use it for building a new CNN, e.g. using a U-Net. All you need is:
 - A main script calling the training and inference
 - A data generator
 - Data
 - A pickle file containing the splits between training and validation
 
### Setup local training folder
Two templates are available for main.py and data_generator.py from the github repo:
```
mkdir your_project
cp CAAI/CNN/scripts/* your_project
```
You need to edit in both files to match your project.

In the same script folder, an example file for the data split is available (generate_data_pickle.py).

### Training
Once you have set up your data, data_generator and main functions, you are ready to train - just run 
```
python3 main.py
```

The script will generate 
 - TensorBoard logs in "logs"
 - Checkpoint models in "checkpoint"
 - Config files in "configs"
 
### Resume model training
If your training is interrupted, you can resume from last saved checkpoint by just running the training 
again, it will automatically resume training if you did not modify any of the parameters in the config.

## HOW TO INSTALL
```
mkdir build
cd build
cmake ..
make install
```
## POST INSTALLATION
Add "source /opt/caai/toolkit-config.sh" to .bashrc / .bash_profile 

## KNOWN ISSUES

### Install on ubuntu
sudo make install gives error on install_manifest.txt

#### Solution:
Prior to install run:
```
sudo mkdir /opt/caai
sudo chown -R <user>:<group> /opt/caai
```
Proceed to install with "make install" without sudo.

## Examples for usage

### hello-world example
Example setting only model name, data directory and train/valid splits.
```
cnn = CNN(model_name='v1',
          data_pickle='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc/data_6fold.pickle',
          data_folder='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc'
      )
cnn.data_loader = DataGenerator(cnn.config)  

cnn.train()    
```

### Custom network and metrics example
Use of custom network architecture and metrics imported from elsewhere.

```
from networks import customNET
from CAAI.losses import rmse

cnn = CNN(model_name='v1',
          data_pickle='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc/data_6fold.pickle',
          data_folder='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc'
      )
cnn.data_loader = DataGenerator(cnn.config)  

cnn.custom_network_architecture = customNET

cnn.metrics.append('mse')
cnn.metrics.append(rmse)
cnn.custom_objects['rmse'] = rmse
cnn.compile_network()

cnn.train()    
```

### Custom loss
Use of custom losses

```
from networks import customNET
from CAAI.losses import rmse

cnn = CNN(model_name='v1',
          data_pickle='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc/data_6fold.pickle',
          data_folder='/users/claes/projects/LowdosePET/PETrecon/HjerteFDG_mnc'
      )
cnn.data_loader = DataGenerator(cnn.config)  

cnn.loss_functions=[[rmse,1]]
cnn.custom_objects['rmse'] = rmse
cnn.compile_network()

cnn.train()    
```

#### github test / DGK
    