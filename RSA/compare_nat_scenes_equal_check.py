from compare_reps_equal_check import *

PATH_CPC_mousenet = "/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_4_gpu_b_1024_lr_005/model_best.pth.tar"



R = compare_reps(model_type = [('CPC','mousenet', PATH_CPC_mousenet)], 
             StimType = 'natural_scenes', 
             area = [('VISp',275,'Cux2-CreERT2')], 
             path_fig = '/nfs/gatsbystor/ammarica/RSA/checks/equal_check',
             compare_models = True,
             plot_hierarchy = False)