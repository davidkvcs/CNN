project (rh-scripts)

cmake_minimum_required (VERSION 2.8)

#/opt/bin was changed to /homes/kovacs/toolbox for local installation
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "/homes/kovacs/toolbox" CACHE PATH "default install path" FORCE )
endif()

# Add python-lib
ADD_SUBDIRECTORY( pythontoolkit )

# Install programs into bin
INSTALL(PROGRAMS bin/merge_tensorboard_logs.py
  PERMISSIONS
    OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ  WORLD_EXECUTE WORLD_READ
  DESTINATION
    bin )
    
    
# Install PATH and PYTHONPATH configuration file
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/toolkit-config.sh.cmake ${CMAKE_CURRENT_BINARY_DIR}/toolkit-config.sh @ONLY)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/toolkit-config.sh DESTINATION .)
