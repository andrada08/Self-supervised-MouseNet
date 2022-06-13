# Analysis

from allenDataAnalysis import *

area = 'VISpm'

all_RSM_Allen, noise_ceiling_rsm, noise_ceiling_cka, activations = get_RSM(
                                                                            CreLine = ['Cux2-CreERT2'], TargetedStruct = [area], 
                                                                            ImagingDepth = [175,275], StimType = 'natural_movie_one',downsample_rate = 1/(3*5), 
                                                                            pool_sessions = True
                                                                            )