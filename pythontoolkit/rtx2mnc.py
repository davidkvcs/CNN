# Import libraries:
import numpy as np
import pydicom as dicom
from pydicom.filereader import InvalidDicomError
import pyminc.volumes.factory as pyminc
import argparse
import cv2
import os


# Define function that converts RT struct to binary minc mask:
def rtx2mnc(RTX,MINC,RTMINC,verbose=False,copy_name=False):
    try:
        # Read RT file and load Region of Interests (ROI's) from the header.
        RTSS = dicom.read_file(RTX) 
        ROIs = RTSS.ROIContourSequence
    
        if verbose:
            print(RTSS.StructureSetROISequence[0].ROIName)
            print("Found",len(ROIs),"ROIs")
        
        # Load input minc file.
        volume = pyminc.volumeFromFile(MINC)
    
        # Convert each ROI into a binary mask.
        for ROI_id,ROI in enumerate(ROIs):
    
            # Create one MNC output file per ROI
            RTMINC_outname = RTMINC if len(ROIs) == 1 else RTMINC[:-4] + "_" + str(ROI_id) + ".mnc"
            RTMINC = pyminc.volumeLikeFile(MINC,RTMINC_outname)
            contour_sequences = ROI.ContourSequence
    
            if verbose:
                print(" --> Found",len(contour_sequences),"contour sequences for ROI:",RTSS.StructureSetROISequence[ROI_id].ROIName)
    
            for contour in contour_sequences:
                assert contour.ContourGeometricType == "CLOSED_PLANAR" # The code is only made to handle contours of this type.
                        
                if verbose:
                    print("\t",contour.ContourNumber,"contains",contour.NumberOfContourPoints)
                
                # Load contour coordinate points into numpy array.
                world_coordinate_points = np.array(contour.ContourData)
                world_coordinate_points = world_coordinate_points.reshape((contour.NumberOfContourPoints,3))
                
                # Convert world (dicom spatial coordinates) coordinates to voxel coordinates.
                voxel_coordinates_inplane = np.zeros((len(world_coordinate_points),2))
                current_slice_i = 0
                for wi,world in enumerate(world_coordinate_points):
                    voxel = volume.convertWorldToVoxel([-world[0],-world[1],world[2]])
                    current_slice_i = voxel[0]
                    voxel_coordinates_inplane[wi,:] = [voxel[2],voxel[1]]
                current_slice_inner = np.zeros((volume.getSizes()[1],volume.getSizes()[2]),dtype=np.float)
                converted_voxel_coordinates_inplane = np.array(np.round(voxel_coordinates_inplane),np.int32)
                
                # Fill out holes in polygon.
                cv2.fillPoly(current_slice_inner,pts=[converted_voxel_coordinates_inplane],color=1)
                
                # Fill out new minc file.
                RTMINC.data[int(round(current_slice_i))] += current_slice_inner                    
    
            # Remove even areas - implies a hole.
            # If a mask is ontop of a larger mask, it is defining a hole in the larger mask. As each mask is binary, the 
            # resulting voxel value will be 2 (or even).
            RTMINC.data[RTMINC.data % 2 == 0] = 0
    
            # Save file.
            RTMINC.writeFile()
            RTMINC.closeVolume()
    
            if copy_name:
                print('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+RTSS.StructureSetROISequence[ROI_id].ROIName+'" '+RTMINC_outname)
                os.system('minc_modify_header -sinsert dicom_0x0008:el_0x103e="'+RTSS.StructureSetROISequence[ROI_id].ROIName+'" '+RTMINC_outname)
    
        volume.closeVolume()
    
    except InvalidDicomError:
        print("Could not read DICOM RTX file",RTX)
        exit(-1)


# Parse input arguments:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RTX2MNC.')
    parser.add_argument('--RTX', help='Path to the DICOM RTX file', type=str)
    parser.add_argument('--MINC', help='Path to the MINC container file', type=str)
    parser.add_argument('--RTMINC', help='Path to the OUTPUT MINC RT file', type=str)
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--copy_name", help="Copy the name of the RTstruct (defined in Mirada) to the name of the MNC file", action="store_true")

    args = parser.parse_args()
    
    # Call function using input arguments.
    rtx2mnc(args.RTX,args.MINC,args.RTMINC,args.verbose,args.copy_name)
        

