
import SimpleITK as sitk
import numpy as np
from ToolsFunc import *
from func_cuRadiomics import func_cuRadiomics


arr_img = np.random.randint(0, high=256, size=(600, 240, 240), dtype='l')
arr_msk = np.ones(arr_img.shape).astype('int')
yaml_addr = 'params.yaml'
features = func_cuRadiomics(yaml_addr, arr_img, arr_msk)

print("1")
