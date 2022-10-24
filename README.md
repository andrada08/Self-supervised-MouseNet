# Self-supervised-MouseNet

The aim of this project was to create a better model of the mouse visual system. 

We did this by using a biologically inspired architecture called MouseNet that has been published [here](https://doi.org/10.1371/journal.pcbi.1010427) with code available [here](https://github.com/mabuice/Mouse_CNN). 

We used self-supervised learning algorithms for training the model. The two implementations tried are [SimSiam](https://doi.org/10.48550/arXiv.2011.10566) and [SimCLR](https://doi.org/10.48550/arXiv.2002.05709). The code used for training is based on the SimSiam code available [here](https://github.com/facebookresearch/simsiam). 

For evaluating the models we used Representational Similarity Analysis (RSA) based on [this](https://github.com/ShahabBakht/ventral-dorsal-model/tree/75f1940bd023dfbafa87bd290bbfaacd2e2b4e85) repo. 




