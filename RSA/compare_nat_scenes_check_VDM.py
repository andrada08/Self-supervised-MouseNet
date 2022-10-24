from compare_reps import *

PATH_CPC_model = "/nfs/nhome/live/ammarica/ventral-dorsal-model/Checkpoints/cpc_p1.pth.tar"

R = compare_reps(model_type = [('CPC','monkeynet_p1', PATH_CPC_model)], 
             StimType = 'natural_scenes', 
             area = [('VISp',275,'Cux2-CreERT2')], 
             path_fig = '/nfs/gatsbystor/ammarica/RSA/checks/model_checkpoints/',
             compare_models = True,
             plot_hierarchy = False)
