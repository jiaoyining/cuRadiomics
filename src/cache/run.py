import numpy as np
import SimpleITK as sitk
import pickle
import glob
import os
import scipy.ndimage
import gzip
from joblib import Parallel, delayed
import random
import cv2
import sys
import radiomics
import six
from radiomics import glcm,glrlm,glszm,gldm,ngtdm

def load_dicom_series(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(os.path.join(path, 'ST0/SE0'))
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    patientID = reader.GetMetaData(0, "0010|0020")

    # z, y, x transpose to x, y, z
    image_array = np.transpose(sitk.GetArrayFromImage(image))
    #image_array = sitk.GetArrayFromImage(image)
    spacing = np.array(image.GetSpacing())

    return patientID, image_array, spacing

def delete_extra_images(origin, mask):

    mask_dim = mask.shape[2]
    z_list = []

    for i in range(mask_dim):
        if np.sum(mask[:, :, i]) != 0:
            z_list.append(i)

    new_origin = rebuild_image(origin, z_list)
    new_mask = rebuild_image(mask, z_list)

    return new_origin, new_mask


def rebuild_image(image, z_list):
    img_list_2d = []

    for z in z_list:
        img_list_2d.append(image[:, :, z])

    return np.transpose(np.array(img_list_2d))

def load_mask(path):
    itk_image = sitk.ReadImage(os.path.join(path, 'mask.nii.gz'))
    image_array = np.transpose(sitk.GetArrayFromImage(itk_image))
    spacing = np.array(itk_image.GetSpacing())

    return image_array, spacing

def main():
    path = '/home/haoxiaoyu/data/cervix_cancer/highly_differentiated/PA0'

    _,image_arr,_ = load_dicom_series(path)

    #mask_arr = np.zeros(image_arr.shape) + 1
    mask_arr,_ = load_mask(path)
    image_arr, mask_arr = delete_extra_images(image_arr, mask_arr)

    image = sitk.GetImageFromArray(image_arr)
    mask = sitk.GetImageFromArray(mask_arr)
    
    feature = glcm.RadiomicsGLCM(image,mask)
    feature.enableAllFeatures()  
    feature.execute()

    for (key,val) in six.iteritems(feature.featureValues):
        print(key,val)
    

if __name__ == "__main__":
    main()
