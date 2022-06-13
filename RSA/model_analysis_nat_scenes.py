from deepModelsAnalysis import *

model_type = 'CPC' 
backbone = 'mousenet' 
pretrained = True

PATH_CPC_mousenet = "/nfs/nhome/live/ammarica/SimSiam-with-MouseNet/Checkpoints/checkpoint_0009.pth.tar"

all_RSM_CPC, activations_CPC, model = get_CPC_RSMs(
                                                    StimType = 'natural_scenes',
                                                    backbone = backbone,
                                                    pretrained =pretrained,
                                                    path=PATH_CPC_mousenet,
                                                    frame_per_block = 5, 
                                                    ds_rate=3
                                                    ) 