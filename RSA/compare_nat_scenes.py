from compare_reps import *

PATH_CPC_mousenet = "/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/Checkpoints/checkpoint_0009.pth.tar"

R = compare_reps(model_type = [('CPC','mousenet', PATH_CPC_mousenet)], 
             StimType = 'natural_scenes', 
             area = [('VISp',275,'Cux2-CreERT2')], 
             path_fig = '/nfs/gatsbystor/ammarica/allendata/RSM_model/figs',
             compare_models = True,
             plot_hierarchy = False)