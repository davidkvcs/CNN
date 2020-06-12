
cmake_minimum_required(VERSION 2.8)


INSTALL(PROGRAMS

  __init__.py
  version.py  
  DataAugmentation3D.py
  losses.py
  networks.py
  train.py
  predict.py
  preprocess_dicom.py
  rtx2mnc.py
  
  DESTINATION CAAI )

