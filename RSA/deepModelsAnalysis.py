import os, sys
from generate_SSM import *
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import scipy as sp
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections
import pdb

data_dir = '/nfs/gatsbystor/ammarica/allendata/RSM_model' #MAKE SURE THIS IS ON GATSBYSTOR
manifest_dir = '/nfs/gatsbystor/ammarica/allendata/RSM_model/boc' #MAKE SURE THIS IS ON GATSBYSTOR

# changed both to only do natural scenes
def prePareAllenStim_for_CPC(exp_id,blocksize,ds_rate):
    boc = BrainObservatoryCache(manifest_file=manifest_dir+'manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    
    scenes = data_set.get_stimulus_template('natural_scenes')
   
    scenes = np.pad(scenes,((0,0),(128,128),(0,0)))

    numImages = scenes.shape[0]
    data = np.ndarray((1,numImages,3,blocksize,64,64))
    for n in range(0,numImages):
        for b in range(0,blocksize):
            thisImage = np.array(scenes[n,:,:])
            thisImage = cv2.resize(thisImage,(64,64)) #
            
            MIN = thisImage.min()
            MAX = thisImage.max()
            thisImage = (thisImage - MIN)/MAX
            thisImage_R = (thisImage + 0.485) * 0.229
            thisImage_G = (thisImage + 0.456) * 0.224
            thisImage_B = (thisImage + 0.406) * 0.225
            
            data[0,n,0,b,:,:] = thisImage_R
            data[0,n,1,b,:,:] = thisImage_G
            data[0,n,2,b,:,:] = thisImage_B
    
    data_colored = data 
       
    return data_colored


def prePareAllenStim_for_othermodels(exp_id, frame_per_block, ds_rate):
    boc = BrainObservatoryCache(manifest_file=manifest_dir+'manifest.json')
    data_set = boc.get_ophys_experiment_data(exp_id)
    
    scenes = data_set.get_stimulus_template('natural_scenes')
    
    numImages = scenes.shape[0]
    data = np.ndarray((numImages,1,224,224))
    for n in range(0,numImages):
        thisImage = np.array(scenes[n,:,0:918])
        thisImage = cv2.resize(thisImage,(224,224))
        thisImage = (thisImage - np.mean(thisImage))/np.std(thisImage)
        data[n,0,:,:] = thisImage
    
    data_colored = np.concatenate((data,data,data),axis=1)
  
    return data_colored
     
def get_activations_CPC(PATH,dataset,backbone,pretrained = True):

    #sys.path.append('../../ventral-dorsal-model/Models/CPC/backbone')
    #sys.path.append('../../ventral-dorsal-model/Models/CPC/dpc')
    #sys.path.append('../SimSiam-with-MouseNet/simsiam')
    #sys.path.append('../../ventral-dorsal-model/RSM')
    #from resnet_2d3d import neq_load_customized
    #from model_3d import DPC_RNN
    #from convrnn import ConvGRUCell
    #from mousenet_complete_pool import Conv2dMask, MouseNetCompletePool
    #from monkeynet import SymmetricConv3d

    curr_wd = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_dir)
    sys.path.append(os.path.join(os.getcwd(),'../../ventral-dorsal-model/Models/CPC/backbone/'))
    sys.path.append(os.path.join(os.getcwd(),'../../ventral-dorsal-model/Models/CPC/dpc/'))
    sys.path.append(os.path.join(os.getcwd(),'../../ventral-dorsal-model/RSM/'))
    sys.path.append(os.path.join(os.getcwd(),'../SimSiam-with-MouseNet/simsiam/'))
    os.chdir(curr_wd)

    from resnet_2d3d import neq_load_customized
    from model_3d import DPC_RNN
    from convrnn import ConvGRUCell
    from mousenet_complete_pool import Conv2dMask, MouseNetCompletePool
    from monkeynet import SymmetricConv3d

    '''
    
	PATH: path to a saved pretrained model
	batch: numpy array of images/stimuli of size (batch size X number of blocks X colors X number of frames X height X width)
	Output: a dictionary containing layer activations as tensors
    
	'''
    model = DPC_RNN(sample_size=64,#128,#48, 184
                        num_seq=60,#24,#120,#5,#8 
                        seq_len=5, 
                        network=backbone,
                        pred_step=3) #3
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

    model.eval()
    if pretrained is True:
        checkpoint = torch.load(PATH,map_location=torch.device('cpu'))
        # here added model.encoder.state_dict()
        
        allkeys = list(model.state_dict())

        pretrained_dict = {}
        for k,v in checkpoint['state_dict'].items():
            newkey = k[15:]
            pretrained_dict[newkey] = v
        
        #model_dict = model.state_dict()
        #model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    activations = collections.defaultdict(list)
	
    if backbone == 'mousenet':
        #if type(model.backbone) is MouseNetGRU:
            # for area in all_mousenet_areas:
            #     print(area)
            #     area_list = [area]
            #     [B,N,C,SL,W,H] = dataset[0].shape
            #     x = dataset[0].view((B*N,C,SL,W,H))
            #     x = x.permute(0,2,1,3,4).contiguous()
            #     re = model.backbone.get_img_feature_no_flatten(x,area_list)
            #     activations[area] = re.detach().numpy()
        # else:

        for area in all_mousenet_areas:
            print(area)
            area_list = [area]
            [B,N,C,SL,W,H] = dataset[0].shape
            x = dataset[0].view((B*N,C,SL,W,H))
            x = x.permute(0,2,1,3,4).contiguous().view((B*N*SL,C,H,W)) 
            #re = model.backbone.get_img_feature_no_flatten(x,area_list)
            re = model.backbone.get_img_feature(x, area_list, flatten=False)
            activations[area] = re.detach().numpy()

    elif backbone == 'simmousenet':

        [B,N,C,SL,W,H] = dataset[0].shape
        x = dataset[0].view((B*N,C,SL,W,H))
        _, re = model.backbone.get_img_features(x)
        for key, value in re.items():
            if key in ['Retina','LGN','VISp_L4','VISal_L4','VISam_L4','VISl_L4','VISpm_L4']:
                BN_l4, C_l4, W_l4, H_l4 = value.shape
                value = value.view((BN_l4//SL, SL, C_l4, W_l4, H_l4))
            print(key, value.shape)
            activations[key] = value.detach().numpy()
        
        
    else:    
        weight_mean = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, ConvGRUCell) or isinstance(m, nn.MaxPool2d) or isinstance(m, SymmetricConv3d) or isinstance(m, nn.MaxPool3d):
                print(f'layer {name}, type {type(m)}')
                # partial to assign the layer name to each hook
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                    weight_mean.append(m.weight.mean())

                m.register_forward_hook(partial(save_activation, activations, name))
        print(f'mean weight = {sum(weight_mean)/len(weight_mean)}')
        with torch.no_grad():
            for batch in dataset:
                print('batch shape: ', batch.shape)
                out = model(batch)

            activations = {name: torch.cat(outputs, 0).detach() for name, outputs in activations.items()}
        for key,value in activations.items():
            activations[key] = value.detach().numpy()        
            
    return activations, model


def save_activation(activations, name, mod, inp, out):#save_activation(name, mod, inp, out,activations):
    activations[name].append(out.cpu().detach())	

def get_activations_othermodels(data_,ModelName):
    
    if ModelName == 'alexnet':
        net = tmodels.alexnet(pretrained=True)
    elif ModelName == 'vgg16':
        net = tmodels.vgg16(pretrained=True)
    elif ModelName == 'resnet18':
        net = tmodels.resnet18(pretrained=False).eval()
        
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    for name, m in net.named_modules():
        if type(m)==nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    # forward pass through the full dataset
    for batch in data_:
        out = net(batch)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    for name in activations.keys():
        activations[name] = activations[name].detach()
    
    return activations, net

def get_othermodels_RSMs(StimType,ModelName,frame_per_block=5,ds_rate=3):
    
    if StimType == 'static_gratings':
        data = prePareVisualStim_for_othermodels()
    
    elif StimType == 'natural_scenes':    
        data = prePareAllenStim_for_othermodels(501498760, frame_per_block, ds_rate)
        
    elif StimType == 'natural_movies':
        _, data = prePareAllenStim_for_othermodels(501498760, frame_per_block, ds_rate)
    
    dataset = [(torch.Tensor(data[:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
    activations, model = get_activations_othermodels(dataset,ModelName)
    activations_centered = center_activations(activations)
    
    all_RSM = compute_similarity_matrices(activations_centered)
    
    return all_RSM, activations, model

       
def get_CPC_RSMs(StimType,backbone,pretrained=False,path='',frame_per_block=5,ds_rate=3):
    

    if StimType == 'drifting_gratings':
        data_DG, _, _ = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_DG[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
        
    elif StimType == 'static_gratings':
        _, data_SG, _ = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_SG[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
    
    elif StimType == 'rdk':
        _, _, data_RDK = prePareVisualStim_for_CPC(5)
        dataset = [(torch.Tensor(data_RDK[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
    
    elif StimType == 'natural_scenes':
        data = prePareAllenStim_for_CPC(501498760,frame_per_block,ds_rate)
        dataset = [(torch.Tensor(data[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
        
    elif StimType == 'natural_movies':
        _, data = prePareAllenStim_for_CPC(501498760,frame_per_block,ds_rate)
        dataset = [(torch.Tensor(data[:,:,:,:,:,:])) for _ in range(1)] # B x N x C x T x W x H
        activations, model = get_activations_CPC(path,dataset,backbone,pretrained=pretrained)
        activations_centered = center_activations(activations)
        all_RSM = compute_similarity_matrices(activations_centered)
        del dataset
   
    
    
    return all_RSM, activations, model
