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

data_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\Bernstein\Data']

figs_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\Bernstein\Plots', r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\Bernstein\Plots_grouped']

for i in range(len(figs_path)):
    if not os.path.isdir(figs_path[i]):
        os.mkdir(figs_path[i])

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

all_options = os.listdir(data_path[0])

all_r = np.empty((len(all_mousenet_areas), len(all_allen_areas), len(all_options)))

for idx_option in range(len(all_options)):

    option = all_options[idx_option]
    option_dir = os.path.join(data_path[0], option)
    # all_noise_ceiling_option = 

    for idx_area in range(len(all_allen_areas)):
        area = all_allen_areas[idx_area]
        this_area_dir = os.path.join(option_dir, area)
        if os.path.isdir(this_area_dir):
            save_path = f'{this_area_dir}\data.hdf5'
            save_file = h5py.File(save_path, 'r')
            r = save_file['r']
            noise_ceiling_rsm = r.attrs['noise_ceiling']
            median_val_r = np.median(r,1)/np.median(noise_ceiling_rsm)
            all_r[:,idx_area,idx_option] = median_val_r 
        if area == 'VISl' and option==all_options[-1]:
            tmp = np.median(r,1)
            tmp2 = np.median(noise_ceiling_rsm)
            print(tmp[-3])
            print('noise:', tmp2)
            print('median: ', median_val_r[-3])
            #print(r[-3, :])
            #missing_r = np.isnan(r[-3,:])
            #print(missing_r)

#missing_r = np.argwhere(np.isnan(all_r))
#print(missing_r)

# plot
sns.set_theme()
for idx_option in range(len(all_options)): 
    option = all_options[idx_option]
    plt.figure()
    ax1 = sns.heatmap(all_r[:,:,idx_option],vmin=-0.1, vmax=0.3, xticklabels=all_allen_areas, yticklabels=all_mousenet_areas)
    figure_name = os.path.join(figs_path[0],all_options[idx_option])
    plt.savefig(figure_name)
    plt.close()


# group by area name
grouped_areas = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISli', 'VISpl', 'VISpor']

# find indices for each area name
tmp_areas_indices = []
for this_area in grouped_areas: 
    indices = [i for i, s in enumerate(all_mousenet_areas) if this_area in s]
    tmp_areas_indices.append(indices)

# get rid of repeats
all_areas_indices = []
for i in range(len(grouped_areas)): 
    indices = tmp_areas_indices[i]
    other_indices = tmp_areas_indices[i+1:]
    flat_other_indices = [x for xs in other_indices for x in xs]
    new = [x for x in indices if x not in flat_other_indices]
    all_areas_indices.append(new)
#print(all_areas_indices)

# group together values from r
grouped_r = np.empty((len(grouped_areas), len(all_allen_areas), len(all_options)))
for i in range(len(all_areas_indices)):
    indices = all_areas_indices[i]
    this_area_r = all_r[indices,:,:]
    grouped_r[i,:,:] = np.nanmean(this_area_r, 0)

# plot
sns.set_theme()
for idx_option in range(len(all_options)): 
    option = all_options[idx_option]
    plt.figure()
    ax1 = sns.heatmap(grouped_r[:,:,idx_option],vmin=-0.1, vmax=0.3, xticklabels=all_allen_areas, yticklabels=grouped_areas)
    figure_name = os.path.join(figs_path[1],all_options[idx_option])
    plt.savefig(figure_name)
    plt.close()

#print(grouped_r[-1,1,-1])