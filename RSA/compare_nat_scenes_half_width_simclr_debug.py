from compare_reps import *

PATH_CPC_mousenet = "/nfs/gatsbystor/ammarica/SimCLR-with-MouseNet/Checkpoints_simclr_half_width_lr_025/model_best.pth.tar"



R = compare_reps(model_type = [('CPC','mousenet', PATH_CPC_mousenet)], 
             StimType = 'natural_scenes', 
             area = [('VISp',275,'Cux2-CreERT2')], 
             path_fig = '/nfs/gatsbystor/ammarica/RSA/figs_half_width_simclr/',
             compare_models = True,
             plot_hierarchy = False)