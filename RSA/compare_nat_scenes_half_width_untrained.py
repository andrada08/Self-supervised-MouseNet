from compare_reps import *

PATH_CPC_mousenet = "/nfs/gatsbystor/ammarica/SimSiam-with-MouseNet/Checkpoints_supervised_half_width_b_1024_lr_05/untrained.pth.tar"



R = compare_reps(model_type = [('CPC','mousenet', PATH_CPC_mousenet)], 
             StimType = 'natural_scenes', 
             area = [('VISp',275,'Cux2-CreERT2'), 
             ('VISl',275,'Cux2-CreERT2'), 
             ('VISal',275,'Cux2-CreERT2'), 
             ('VISpm',275,'Cux2-CreERT2'), 
             ('VISrl',275,'Cux2-CreERT2'), 
             ('VISam',275,'Cux2-CreERT2')], 
             path_fig = '/nfs/gatsbystor/ammarica/RSA/figs_half_width_untrained/',
             compare_models = True,
             plot_hierarchy = False)