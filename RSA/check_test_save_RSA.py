import os
import h5py
import numpy as np

path_fig = '/nfs/gatsbystor/ammarica/RSA/test/'
this_area = 'area'

path_figure = path_fig + this_area + '/'

save_path = f'{path_figure}data.hdf5'
save_file = h5py.File(save_path, 'r')

print(list(save_file.keys()))

R = save_file['R']

print('Shape: ', R.shape)
print('Dtype: ', R.dtype)
print(list(R.attrs.items()))
