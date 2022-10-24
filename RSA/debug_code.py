import argparse
import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#parser = argparse.ArgumentParser(description='Data for RSA plots')
#parser.add_argument('--data_path', help='path to data')
#parser.add_argument('--save_path', help='path to figures')

#args = parser.parse_args()

data_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\RSA_stuff\Data\debug\simclr_half_width\data_VISli2_3.hdf5', 
r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\RSA_stuff\Data\debug\simclr_half_width\data_VISli5.hdf5']



all_allen_areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISrl', 'VISam']
all_mousenet_areas = [#'LGNv',
                          'VISp4',
                          'VISp2/3',
                          'VISp5',
                          'VISal4','VISpl4','VISli4','VISrl4','VISl4',
                          'VISal2/3','VISpl2/3','VISli2/3','VISrl2/3','VISl2/3',
                          'VISal5','VISpl5','VISli5','VISrl5','VISl5',
                          'VISpor4',
                          'VISpor2/3',
                          'VISpor5']

for i in range(len(data_path)):
    save_path = data_path[i]
    print(save_path)
    save_file = h5py.File(save_path, 'r')
    r = save_file['r']
    conv1 = save_file['conv1']
    all_RSM_Allen = save_file['all_RSM_Allen']
    print('conv1 is: ', conv1)
    print('all_RSM_Allen is: ', all_RSM_Allen)