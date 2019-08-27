
import SimpleITK as sitk
import numpy as np
from python.ToolsFunc import *
from python.func_cuRadiomics import func_cuRadiomics
import time
import radiomics


arr_img = np.random.randint(-256, high=256, size=(620, 240, 240), dtype='l')
arr_img[arr_img <= 0] = -1
arr_msk = np.ones(arr_img.shape).astype('int')
yaml_addr = './python/params.yaml'
time_start = time.clock()
features = func_cuRadiomics(yaml_addr, arr_img, arr_msk)
time_end = time.clock()

time_start = time.clock()

features = func_cuRadiomics(yaml_addr, arr_img, arr_msk)
time_end = time.clock()

print('GPU:' + str(time_end - time_start))
