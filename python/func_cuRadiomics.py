import yaml
import numpy as np
import SimpleITK as sitk
from ToolsFunc import *
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time


def func_cuRadiomics(yaml_addr, image, mask):
    tfe.enable_eager_execution()
    a = tf.constant(value=1)
    _RadiomicsGLCMModule = tf.load_op_library('./build/libRadiomics.so').radiomics

    features_name_glcm = ["Autocorrelation",
                          "JointAverage",
                          "ClusterProminence",
                          "ClusterShade",
                          "ClusterTendency",
                          "Contrast",
                          "Correlation",
                          "DifferenceAverage",
                          "DifferenceEntropy",
                          "DifferenceVariance",
                          "JointEnergy",
                          "JointEntropy",
                          "Imc1",
                          "Imc2",
                          "Idm",
                          "Idmn",
                          "Id",
                          "Idn",
                          "InverseVariance",
                          "MaximumProbability",
                          "SumAverage",
                          "SumEntropy",
                          "SumSquares"
                          ]
    features_name_firstorder = \
        ["Energy",
         "Entropy",
         "Minimum",
         "TenthPercentile",
         "NintiethPercentile",
         "Maximum",
         "Mean",
         "Median",
         "InterquartileRange",
         "Range",
         "MAD",
         "rMAD",
         "RMS",
         "StandardDeviation",
         "Skewness",
         "Kurtosis",
         "Varianc",
         "Uniformity"]


    # Reading Parameters if Feature Extraction
    f = open(yaml_addr)
    parameters = yaml.load(f)
    #Range = parameters['Range']
    FirstOrder = parameters['FirstOrder']
    GLCM = parameters['GLCM']
    label = parameters['label']



    # arr_shape = normed_arr_img0.shape
    Range = [np.min(image).astype('int'), np.max(image).astype('int')]
    SETTING = np.array([Range[0], Range[1], FirstOrder, GLCM, label])

    #arr_shape = image.shape
    image[np.where(mask != label)] = -1

    arr_features = _RadiomicsGLCMModule(image, SETTING)
    NumOfFeatures = 0
    Names = []
    if GLCM == 1:
        Names = Names + features_name_glcm
        NumOfFeatures += 23
    if FirstOrder == 1:
        Names = Names + features_name_firstorder
        NumOfFeatures += 18

    arr_features = np.reshape(arr_features, newshape=[NumOfFeatures, image.shape[0]])

    RadiomicsFeatures = {}
    for i in range(len(Names)):
        RadiomicsFeatures[Names[i]] = arr_features[i, :]

    return RadiomicsFeatures

