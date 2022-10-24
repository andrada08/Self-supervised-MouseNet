from compare_reps import *


R = compare_reps(model_type = [('VGG', 'resnet18')], 
             StimType = 'natural_scenes', 
             area = [('VISp',275,'Cux2-CreERT2'), 
             ('VISl',275,'Cux2-CreERT2'), 
             ('VISal',275,'Cux2-CreERT2'), 
             ('VISpm',275,'Cux2-CreERT2'), 
             ('VISrl',275,'Cux2-CreERT2'), 
             ('VISam',275,'Cux2-CreERT2')], 
             path_fig = '/nfs/gatsbystor/ammarica/RSA/figs_resnet18/untrained/',
             compare_models = True,
             plot_hierarchy = False)