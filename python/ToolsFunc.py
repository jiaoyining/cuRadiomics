import SimpleITK as sitk
import six
import sys, os
import radiomics
import numpy as np
import csv
import pandas as pd
import matplotlib as plt


def write_csv_features_cu(features_name, name_feature_file, name_subj, values, start):

    num_of_slices = values.shape[0]
    labels = ('flair', 't1', 't1ce', 't2')

    with open(name_feature_file, 'a') as f:
        if start == 1:
            f.write('name' + ',')
            for key in features_name:
                f.write(str(key) + ',')
            f.write('\n')
        for i_slice in range(num_of_slices):

            f.write(str(name_subj) + '_' + str(i_slice) + ',')
            for i_value in values[i_slice, :]:
                f.write(str(i_value) + ',')
            f.write('\n')

    return

def write_csv_features(dict_1, file_1, name_subj, start):
    with open(file_1, 'a+') as f:
        # the head of the result file
        if start == 1:
            f.write('name' + ',')
            for key, value in list(dict_1.items()):
                f.write(str(key) + ',')
            f.write('\n')
        # writing values
        f.write(str(name_subj) + ',')
        for key, value in list(dict_1.items()):
            f.write(str(value) + ',')
        f.write('\n')
        f.close()


def write_csv_times(name_time_file, labels, dict_time):

    with open(name_time_file, 'a') as f:
        for i_label in labels:
            f.write(str(i_label) + ',' + str(dict_time[i_label]))
            f.write('\n')
    return


def startWith(dict_1, file_1):


    with open(file_1, 'a') as f:
        for key, value in list(dict_1.items()):
            f.write(str(key) + ',')
            f.write(str(value) + '\n')

    def run(s):
        f = map(s.startswith, starts)
        if True in f: return s

    return run


def endWith(*endstring):
    ends = endstring

    def run(s):
        f = map(s.endswith, ends)
        if True in f: return s

    return run


def write_csv(dict_1, file_1):
    with open(file_1, 'a') as f:
        for key, value in list(dict_1.items()):
            f.write(str(key) + ',')
            f.write(str(value) + '\n')




def write_csv_with_parameters(dict_1, file_1, feature_name, param_filename):

    par_names = ("ImT", "bin", "maxd", "wN", "smtc", "alpha",)
    len_of_featurename = len(feature_name)
    len_of_filename = len(param_filename)
    cnt = param_filename.count('_')
    par_dic = {}
    END = param_filename.find('.')

    for par in par_names:
        judge = param_filename.find(par)
        if judge == -1:
            par_dic[par] = "-"
        else:
            begin_pos = param_filename[:END].find('_', judge)
            end_pos = param_filename[:END].find('_', begin_pos+1)
            par_dic[par] = param_filename[(begin_pos+1):end_pos]

    parameter_of_feature = param_filename[(len_of_featurename):(len_of_filename-6)]
    with open(file_1, 'a') as f:
        for key, value in dict_1.items()[9:]:
            f.write(str(key) + ',')
            f.write(str(value) + ',')
            for par in par_names:
                f.write(par_dic[par] + ',')
            f.write(str(key) + parameter_of_feature + ',')
            f.write('\n')
    return 0

def single_mask(image, label):
    array = sitk.GetArrayFromImage(image)
    array[array != label] = 0
    array[array == label] = 1
    after_image = sitk.GetImageFromArray(array)
    after_image.CopyInformation(image)
    return after_image


def fuse_mask(image):
    array = sitk.GetArrayFromImage(image)
    array[array > 0] = 1
    after_image = sitk.GetImageFromArray(array)
    after_image.CopyInformation(image)
    return after_image


def select_file(file_address, selected):
    listed_file = os.listdir(file_address)
    selector = endWith(selected) #('_p2.nii')
    file_name = list(filter(selector, listed_file))
    if not len(file_name) == 0:
        file_selected = file_address + file_name[-1]
    else:
        file_selected = ''
    return file_selected


def normalize(img_arr, msk_arr):

    #labelArr=sitk.GetArrayFromImage(label)
    ''''
    min_value = np.percentile(img_arr, 0.1).astype('float')
    max_value = np.percentile(img_arr, 99.9).astype('float')
    img_arr[img_arr > max_value] = max_value
    img_arr[img_arr < min_value] = min_value   #-outliers
    new_arr = (img_arr-min_value)/(max_value-min_value)*scale
    '''
    num_of_slices = img_arr.shape[0]
    new_arr = np.zeros(img_arr.shape)
    img_arr[np.where(msk_arr != 1)] = 0
    for i_slice in range(num_of_slices):
        img_arr[i_slice, :, :] = (img_arr[i_slice, :, :] - img_arr[i_slice, :, :].min()).astype('int32')

        hist, bins = np.histogram(img_arr[i_slice, :, :].flatten(), img_arr[i_slice, :, :].max()+1)
        cdf = hist.cumsum()  # 计算累积直方图
        cdf_m = np.ma.masked_equal(cdf, 0)  # 除去直方图中的0值
        cdf_m = (cdf_m - cdf_m.min()) * bins.max()/ (cdf_m.max() - cdf_m.min())  # 等同于前面介绍的lut[i] = int(255.0 *p[i])公式
        cdf = np.ma.filled(cdf_m, 0).astype('int')

        new_arr[i_slice, :, :] = cdf[img_arr[i_slice, :, :]]
        new_arr[i_slice, :, :] = (new_arr[i_slice, :, :]-new_arr[i_slice, :, :].min())*255/(new_arr[i_slice, :, :].max()-new_arr[i_slice, :, :].min()+1)

    return new_arr.astype('int32')


def order_times(patient_times):

    time_array = np.array(patient_times)
    times_from_previous = np.sort(time_array)

    return times_from_previous


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def resample(t2, adc):
    t2_image = sitk.ReadImage(t2)
    adc_image = sitk.ReadImage(adc)

    t2_size = np.array(list(t2_image.GetSize()))
    adc_size = np.array(list(adc_image.GetSize()))

    t2_spacing = np.array(list(t2_image.GetSpacing()))
    adc_spacing = np.array(list(adc_image.GetSpacing()))

    t2_direction = np.array(list(t2_image.GetDirection()))
    adc_direction = np.array(list(adc_image.GetDirection()))

    t2_org = np.array(list(t2_image.GetOrigin()))
    adc_org = np.array(list(adc_image.GetOrigin()))

    t2_centralPixel = (t2_size - 1.0) / 2
    adc_centralPixel = (adc_size - 1.0) / 2

    sum_adc_1 = adc_org + adc_spacing * adc_centralPixel
    sum_adc_0 = adc_org + adc_spacing * (adc_size - 1 - adc_centralPixel)

    org_1 = sum_adc_1 - (t2_spacing * t2_centralPixel)
    org_0 = sum_adc_0 - t2_spacing * (t2_size - 1 - t2_centralPixel)

    org_1 = totuple(org_1)

    resample = sitk.ResampleImageFilter()
    resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resample.SetReferenceImage(adc_image)
    resample.SetOutputDirection(adc_image.GetDirection())
    resample.SetOutputOrigin(org_1)
    resample.SetOutputSpacing(adc_image.GetSpacing())
    resample.SetSize(adc_image.GetSize())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    imageCA = resample.Execute(t2_image)
    return imageCA


def batch_normalization(image_addr, mask_addr, image_selector, mask_selector, out_addr, result_str):
    # 1.choose standard mask
    #if modality == 'T2':
    #    #mask_name_end = '_t2_prostate.nii.gz'
    #    mask_name_end = selector_str + '.nii.gz'
    #    image_name_end = '_p2.nii'
    #    out_name_end = '_t2_p2.nii.gz'
    #if modality == 'ADC':
    #    #mask_name_end = 'label_prostate.nii.gz'
    #    mask_name_end = selector_str + '.nii.gz'
    #    image_name_end = '_ADC.nii'
    #    out_name_end = '_ADC.nii.gz'

    image_name_end = image_selector + '.nii'
    mask_name_end = mask_selector + '.nii.gz'
    # reading image and mask
    subject = os.listdir(image_addr) #'../data/20170714/')
    for i in subject:
        image_time = image_addr + str(i) + '/'   # '../data/20170714/'
        mask_time = mask_addr + str(i) + '/'
        times = os.listdir(mask_time)

        for j in times:
            image_address = image_time + str(j) + '/'
            mask_address = mask_time + str(j) + '/'
            ImageName = select_file(image_address, image_name_end)  # '_p2.nii')
            ProstateMaskName = select_file(mask_address, mask_name_end)
            prostate_image = sitk.ReadImage(ImageName)
            print('-----' + i + '-----' + j + '-----' + image_selector + '-----' + mask_selector + '-----reading')

            # 2. if mask exists, normalize iamge using this mask
            if os.path.isfile(ProstateMaskName):
                mask = sitk.ReadImage(ProstateMaskName)
                prostate_mask = fuse_mask(mask)
                normalized_image = normalizeImage(prostate_image, prostate_mask, 1, 255)
            # 3.saving normalization results
                out_address = out_addr + str(i) + '/' + str(j) + '/'
                if not os.path.exists(out_address):
                    os.makedirs(out_address)
                out_name = out_address + str(i) + '_' + str(j) + '_Normalized_' + result_str + '.nii.gz'
                sitk.WriteImage(normalized_image, out_name)
                print('-----' + i + '-----' + j + '-----' + image_selector + '-----' + mask_selector + '-----completed')
    return 0


def mask_convert(image_address, image_selector, mask_selector, mask_address, out_str, out_addr):

    subjects = os.listdir(mask_address) #"../data/20170714/"
    image_name_end = image_selector + '.nii'
    mask_name_end = mask_selector + '.nii.gz'

    for i in subjects:
        image_time = image_address + str(i) + "/" # "../data/20170714/"
        mask_time = mask_address + str(i) + "/"
        times = os.listdir(mask_time)

        for j in times:
            image_addr = image_time + str(j) + "/"
            mask_addr = mask_time + str(j) + "/"
            ADC_image_name = select_file(image_addr, image_name_end)

            # 1. convert t2 prostate label to ADC image
            T2_mask_name = select_file(mask_addr, mask_name_end)

            if not len(T2_mask_name) == 0:
                out_address = out_addr + str(i) + '/' + str(j) + '/'
                if os.path.exists(out_address) == 0:
                    os.makedirs(out_address)
                OutName = os.path.join(out_address, str(i) + '_' + str(j) + '_' + out_str + '.nii.gz')
                regis = "plastimatch convert --interpolation nn --input " + str(T2_mask_name) + " --fixed " + str(ADC_image_name) + " --output-img " + OutName
                os.system(regis)

            # 2.load inner and outer label to ADC
            #T2InOutMaskName = select_file(ImageAddress, '_t2_pz_tz.nii.gz')
            #if not len(T2InOutMaskName) == 0:
            #    OutMask = ImageAddress + 'label_center.nii.gz'
            #    regis = "plastimatch convert --interpolation nn --input " + str(T2InOutMaskName) + " --fixed " + str(ADCImageName) + " --output-img " + OutMask
            #    os.system(regis)

    return 0


def generating_no_tumour_mask(prostate, tumour):

    image_prostate = sitk.ReadImage(prostate)
    image_tumour = sitk.ReadImage(tumour)
    arr_prostate = sitk.GetArrayFromImage(image_prostate)
    arr_tumour = sitk.GetArrayFromImage(image_tumour)
    new_arr = np.copy(arr_prostate)
    new_arr[np.where(arr_tumour > 0)] = 0
    new_image = sitk.GetImageFromArray(new_arr)
    new_image.CopyInformation(image_prostate)

    return new_image

def batch_generating(tumour_folder, sav_folder, prostate_selector, tumour_selector, modality):

    patients = os.listdir(tumour_folder)
    patients.sort(key=str.lower)
    PatientName_Existed = []

    prostate_name_end = prostate_selector + '.nii.gz'
    tumour_name_end = tumour_selector + '.nii.gz'
    for i in patients:

        patient_folder = tumour_folder + i + '/'
        times = os.listdir(patient_folder)
        time_ordered = order_times(times)

        for j in time_ordered:
            print('begin---------subject:' + i + '----------, time:' + j + '--------------')
            # make saving folder
            addr = patient_folder + j + '/'
            saving_folder = sav_folder + i + '/' + j + '/'
            if os.path.exists(saving_folder) == 0:
                os.makedirs(saving_folder)

            prostate_name = select_file(addr, prostate_name_end)
            tumour_name = select_file(addr, tumour_name_end)

            name = addr + 'p-t_' + modality + '.nii.gz'
            prostate_without_tumour = generating_no_tumour_mask(prostate_name, tumour_name)
            sitk.WriteImage(prostate_without_tumour, name)
            print('end---------subject:' + i + '----------, time:' + j + '--------------')
    return

