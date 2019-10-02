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