import argparse
import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torchvision.models as tmodels


#parser = argparse.ArgumentParser(description='Data for RSA plots')
#parser.add_argument('--data_path', help='path to data')
#parser.add_argument('--save_path', help='path to figures')

#args = parser.parse_args()

data_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\RSA_stuff\Resnet18\Data\trained', r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\RSA_stuff\Resnet18\Data\untrained']

figs_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\RSA_stuff\Resnet18\Plots']

for i in range(len(figs_path)):
    if not os.path.isdir(figs_path[i]):
        os.mkdir(figs_path[i])

all_allen_areas = ['VISp', 'VISl', 'VISal', 'VISpm', 'VISrl', 'VISam']

# trained
net = tmodels.resnet18(pretrained=True)

all_layers_names = []
for name, m in net.named_modules():
        if type(m)==nn.Conv2d:
            all_layers_names.append(name)

#print(all_layers_names)

grouped_layers = ['layer1', 'layer2', 'layer3', 'layer4']

# find indices for each layer
tmp_areas_indices = []
for this_area in grouped_layers: 
    indices = [i for i, s in enumerate(all_layers_names) if this_area in s]
    tmp_areas_indices.append(indices[-1])

#print(tmp_areas_indices)

# trained
all_r_trained = np.empty((len(grouped_layers), len(all_allen_areas)))
for idx_area in range(len(all_allen_areas)):
        area = all_allen_areas[idx_area]
        this_area_dir = os.path.join(data_path[0], area)
        if os.path.isdir(this_area_dir):
            save_path = f'{this_area_dir}\data.hdf5'
        save_file = h5py.File(save_path, 'r')
        r = save_file['r']
        noise_ceiling_rsm = r.attrs['noise_ceiling']
        median_val_r = np.median(r,1)/np.median(noise_ceiling_rsm)
        all_r_trained[:,idx_area] = median_val_r[tmp_areas_indices]

sns.set_theme()
plt.figure()
ax1 = sns.heatmap(all_r_trained[:,:], vmin=-0.1, vmax=0.3, xticklabels=all_allen_areas, yticklabels=grouped_layers)
figure_name = os.path.join(figs_path[0],'Resnet18_trained')
plt.savefig(figure_name)


# not trained
all_r_untrained = np.empty((len(grouped_layers), len(all_allen_areas)))
for idx_area in range(len(all_allen_areas)):
        area = all_allen_areas[idx_area]
        this_area_dir = os.path.join(data_path[1], area)
        if os.path.isdir(this_area_dir):
            save_path = f'{this_area_dir}\data.hdf5'
        save_file = h5py.File(save_path, 'r')
        r = save_file['r']
        noise_ceiling_rsm = r.attrs['noise_ceiling']
        median_val_r = np.median(r,1)/np.median(noise_ceiling_rsm)
        all_r_untrained[:,idx_area] = median_val_r[tmp_areas_indices]


sns.set_theme()
plt.figure()
ax1 = sns.heatmap(all_r_untrained[:,:], vmin=-0.1, vmax=0.3, xticklabels=all_allen_areas, yticklabels=grouped_layers)
figure_name = os.path.join(figs_path[0],'Resnet18_untrained')
plt.savefig(figure_name)

