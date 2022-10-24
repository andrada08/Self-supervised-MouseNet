import argparse
from enum import unique
import os
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\RSA_stuff\Poster_data']
figs_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\Self-supervised-MouseNet\other']

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
#print(all_options)

# get all values
all_r = np.empty((len(all_mousenet_areas), len(all_allen_areas), len(all_options), 3))
for idx_option in range(len(all_options)):
    option = all_options[idx_option]
    option_dir = os.path.join(data_path[0], option)
    all_width_options = os.listdir(option_dir)
    #print(all_width_options)
    for width_idx in range(len(all_width_options)):
        width_dir = os.path.join(option_dir, all_width_options[width_idx])
        for idx_area in range(len(all_allen_areas)):
            area = all_allen_areas[idx_area]
            this_area_dir = os.path.join(width_dir, area)
            if os.path.isdir(this_area_dir):
                save_path = f'{this_area_dir}\data.hdf5'
                save_file = h5py.File(save_path, 'r')
                r = save_file['r']
                noise_ceiling_rsm = r.attrs['noise_ceiling']
                median_val_r = np.median(r,1)/np.median(noise_ceiling_rsm)
                all_r[:,idx_area,idx_option,width_idx] = median_val_r 


# print(all_r)

# average for each option_width
values_mean_r = np.empty((len(all_options),3))
i = 0
for idx_option in range(len(all_options)):
    option = all_options[idx_option]
    option_dir = os.path.join(data_path[0], option)
    all_width_options = os.listdir(option_dir)
    for width_idx in range(len(all_width_options)):
        values_mean_r[idx_option, width_idx] = np.nanmean(np.nanmean(all_r[:,:,idx_option,width_idx], 0), 0)

# print(values_mean_r)



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
grouped_r = np.empty((len(grouped_areas), len(all_allen_areas), len(all_options), len(all_width_options)))
for i in range(len(all_areas_indices)):
    indices = all_areas_indices[i]
    this_area_r = all_r[indices,:,:,:]
    grouped_r[i,:,:,:] = np.mean(this_area_r, 0)


# plt.figure()
# plt.plot([0.5, 1, 2], np.transpose(values_mean_r))
# ax = plt.gca()
# ax.set_ylim((0.06, 0.12))
# ax.set_xlim((0.4, 2.1))
# ax.legend(all_options)
# plt.savefig(r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\Self-supervised-MouseNet\other\random_mean.png')

# average per option per area 
area_values_mean_r = np.empty((len(all_options), 3, len(all_allen_areas)))
for idx_option in range(len(all_options)):
    option = all_options[idx_option]
    option_dir = os.path.join(data_path[0], option)
    all_width_options = os.listdir(option_dir)
    for width_idx in range(len(all_width_options)):
        for idx_area in range(len(all_allen_areas)):
            area_values_mean_r[idx_option, width_idx, idx_area] = np.nanmean(np.nanmean(np.nanmean(all_r[:,idx_area,idx_option,width_idx], 0), 0), 0)

# for idx_area in range(len(all_allen_areas)):
#     plt.figure()
#     plt.plot([0.5, 1, 2], np.transpose(area_values_mean_r[1:, :, idx_area]))
#     ax = plt.gca()
#     ax.legend(all_options[1:])
#     figure_name = os.path.join(figs_path[0], all_allen_areas[idx_area])
#     plt.savefig(figure_name)
#     plt.close()

# # make big plot
# fig, axs = plt.subplots(3, 2)
# i = 0
# j = 0
# for idx_area in range(len(all_allen_areas)):
#     axs[i,j].plot([0.5, 1, 2], np.transpose(area_values_mean_r[1:, :, idx_area]))
#     axs[i,j].set_title(all_allen_areas[idx_area])
#     j = j + 1
#     if j==2:
#         i = i + 1
#         j = 0

# # axs[0,2].legend(all_options[1:])
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.6, 
#                     hspace=0.6)
# figure_name = os.path.join(figs_path[0], 'subplots_area')
# plt.savefig(figure_name)
# plt.close
    


# # make plot like mousenet paper
# for idx_area in range(len(all_allen_areas)):
#     for idx_width in range(len(all_width_options)):
#         plt.rcParams["figure.figsize"] = [14, 3.50]
#         plt.rcParams["figure.autolayout"] = True
#         fig = plt.figure()

#         for idx_option in range(1, len(all_options)):
#             plt.plot(all_mousenet_areas, all_r[:,idx_area,idx_option,idx_width])

#         plt.legend(all_options[1:])
#         plt.ylim((0, 0.3))
#         figure_name = 'ungrouped_' + all_allen_areas[idx_area] + all_width_options[idx_width]
#         plt.savefig(figure_name)
#         plt.close()


#         plt.rcParams["figure.figsize"] = [10, 3.50]
#         plt.rcParams["figure.autolayout"] = True
#         fig = plt.figure()

#         for idx_option in range(1, len(all_options)):
#             plt.plot(grouped_areas, grouped_r[:,idx_area,idx_option,idx_width])

#         plt.legend(all_options[1:])
#         plt.ylim((0, 0.3))
#         figure_name = 'grouped_' + all_allen_areas[idx_area] + all_width_options[idx_width]
#         plt.savefig(figure_name)
#         plt.close()


import pandas as pd

acc_res_path = [r'C:\Users\Andrada\OneDrive\Documents\Gatsby_code\RSA_stuff\Accuracy_stuff\results_eval.csv']

acc_rs_df = pd.read_csv(acc_res_path[0])
acc_values = acc_rs_df['acc1']
acc_names = acc_rs_df['name']
unique_names = pd.unique(acc_names)

## accuracy vs width
#sns.set_theme(font_scale=2.0, style='whitegrid')
#plt.figure()
#for i in range (3, len(acc_values)-2, 3):
#    plt.plot([0.5, 1, 2], acc_values[i:i+3])
#plt.legend(unique_names[1:])
#plt.xlim([0.4, 2.1])
#plt.savefig('accuracy_random')
#plt.close()

# accuracy vs width
sns.set_theme(font_scale=2.0, style='whitegrid')
plt.figure(figsize=(11, 7), dpi=500)
for i in range (3, len(acc_values)-2, 3):
    plt.plot([0.5, 1, 2], acc_values[i:i+3], linewidth=2.5)
plt.legend(unique_names[1:])
plt.xlim([0.4, 2.1])

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=25)
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
ax.xaxis.set_major_locator(MultipleLocator(0.5))

plt.xlabel('Model scale for width', fontsize=27)
plt.ylabel('Top1 accuracy', fontsize=27)
#make figure great again!!
#plt.rcParams["figure.figsize"] = [400, 300]
#plt.rcParams["figure.autolayout"] = True

plt.savefig('accuracy_random_new')
plt.close()



# accuracy x RSA per area
# for idx_area in range(len(all_allen_areas)):
#     for idx_width in range(len(all_width_options)):
#         plt.figure()
#         for i in range(3, len(acc_values)-2, 3):
#             plt.plot(acc_values[i:i+3], area_values_mean_r[1:, idx_width, idx_area])

#         plt.legend(unique_names[1:])
#         figure_name = 'acc_and_RSA_for_' + all_allen_areas[idx_area] + all_width_options[idx_width]
#         #plt.title(all_allen_areas[idx_area])
#         plt.savefig(figure_name)
#         plt.close()

# make subplots per width
# for idx_width in range(len(all_width_options)):
#     fig, axs = plt.subplots(3, 2)
#     i = 0
#     j = 0
#     for idx_area in range(len(all_allen_areas)):
#         for idx_val in range(len(unique_names[1:])):
#             axs[i,j].plot(acc_values[[idx_val+3,idx_val+6,idx_val+9]], area_values_mean_r[1:, idx_width, idx_area])
#             axs[i,j].set_title(all_allen_areas[idx_area])
#         j = j + 1
#         if j==2:
#             i = i + 1
#             j = 0
#     plt.legend(unique_names[1:])
#     figure_name = 'subplots_acc_and_RSA_for_' + (all_width_options[idx_width])[:-10]
#     #plt.title(all_allen_areas[idx_area])
#     plt.savefig(figure_name)
#     plt.close()

# for idx_option in range(len(all_options)):
#     fig, axs = plt.subplots(3, 2)
#     i = 0
#     j = 0
#     for idx_area in range(len(all_allen_areas)):
#         for idx_val in range(len(unique_names[1:])):
#             axs[i,j].plot(acc_values[idx_val:idx_val+4,idx_val+8]], area_values_mean_r[idx_option, :, idx_area])
#             axs[i,j].set_title(all_allen_areas[idx_area])
#         j = j + 1
#         if j==2:
#             i = i + 1
#             j = 0
#     plt.legend(unique_names[1:])
#     figure_name = 'subplots_acc_and_RSA_for_' + (all_width_options[idx_width])[:-10]
#     #plt.title(all_allen_areas[idx_area])
#     plt.savefig(figure_name)
#     plt.close()

fig, axs = plt.subplots(3, 2, dpi=600)
i = 0
j = 0
for idx_area in range(len(all_allen_areas)):
    sns.set_theme(font_scale=1.2, style='whitegrid')
    axs[i,j].set_title(all_allen_areas[idx_area])
    for idx_val, idx_option in zip(range(3, len(acc_values)-2, 3), range(len(all_options))):
        axs[i,j].plot(acc_values[idx_val:idx_val+3], area_values_mean_r[idx_option, :, idx_area])
    axs[i,j].set_ylim([-0.01, 0.23])
    axs[i,j].set_xlabel('Top1 accuracy', fontsize=10)
    axs[i,j].set_ylabel('RSA score', fontsize=10)
    plt.rcParams["figure.figsize"] = [500, 400]
    plt.rcParams["figure.autolayout"] = True
    j = j + 1
    if j==2:
        i = i + 1
        j = 0
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=1.1)
for a in axs.flatten():
    a.tick_params(axis='both', which='major', labelsize=10)
    
    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

    a.xaxis.set_major_locator(MultipleLocator(5))
    a.yaxis.set_major_locator(MultipleLocator(0.05))

    # Change minor ticks to show every 5. (20/4 = 5)
    a.xaxis.set_minor_locator(AutoMinorLocator(5))
    a.yaxis.set_minor_locator(AutoMinorLocator(0.01))

    # # And a corresponding grid
    # a.grid(which='both')

    # # Or if you want different settings for the grids:
    # a.grid(which='minor', alpha=0.2)
    # a.grid(which='major', alpha=0.5)
#fig.legend(unique_names[1:], loc='upper center')
figure_name = 'subplots_acc_and_RSA'
#plt.title(all_allen_areas[idx_area])
plt.savefig(figure_name)
plt.close()