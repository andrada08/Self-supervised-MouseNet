import os
import h5py
import numpy as np

path_fig = '/nfs/gatsbystor/ammarica/RSA/test/'
this_area = 'area'

path_figure = path_fig + this_area + '/'
if not os.path.isdir(path_figure):
    os.mkdir(path_figure)


this_model_type = 'test_CPC'
this_backbone = 'test_mousenet'


R =  np.arange(100)
noise_ceiling = 0

# save stuff
save_path = f'{path_figure}data.hdf5'
save_file = h5py.File(save_path, 'w')
dset = save_file.create_dataset('R', data=R)
dset.attrs['model'] = this_model_type
dset.attrs['backbone'] = this_backbone
dset.attrs['noise_ceiling'] = noise_ceiling
save_file.close()

print('Saved RSA info')