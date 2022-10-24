import pandas as pd

# define model names and used widths
name = ['Supervised', 'SimCLR', 'Same acc sup', 'Ecoset sup', 'Ecoset simclr']
datasets = ['imagenet', 'ecoset']
width = ['0.5', '1', '2']

# manually put in all paths to desired checkpoints
ecoset_sup_checkpoints = ['/nfs/gatsbystor/ammarica/Ecoset/Checkpoints_supervised_half_width_b_1024_lr_005',
                    '/nfs/gatsbystor/ammarica/Ecoset/Checkpoints_supervised_b_1024_lr_01', 
                    '/nfs/gatsbystor/ammarica/Ecoset/Checkpoints_supervised_2_wider_b_1024_lr_015']

ecoset_simclr_checkpoints = ['/nfs/gatsbystor/ammarica/Ecoset/Checkpoints_simclr_half_width_b_1024_lr_01',
                    '/nfs/gatsbystor/ammarica/Ecoset/Checkpoints_simclr_b_1024_lr_05', 
                    '/nfs/gatsbystor/ammarica/Ecoset/Checkpoints_simclr_2_wider_b_1024_lr_05']

simclr_checkpoints = ['/nfs/gatsbystor/ammarica/SimCLR-with-MouseNet/Checkpoints_simclr_half_width_lr_025',
                    '/nfs/gatsbystor/ammarica/SimCLR-with-MouseNet/Checkpoints_simclr_2_gpu_b_1024_lr_05',
                    '/nfs/gatsbystor/ammarica/SimCLR-with-MouseNet/Checkpoints_simclr_2_wider_lr_075']

same_acc_sup_checkpoints = ['/nfs/gatsbystor/ammarica/Same_acc_sup/Checkpoints_supervised_half_width_b_1024_lr_05_new',
'/nfs/gatsbystor/ammarica/Same_acc_sup/Checkpoints_supervised_b_1024_lr_01',
'/nfs/gatsbystor/ammarica/Same_acc_sup/Checkpoints_supervised_2_wider_b_1024_lr_075']

supervised_checkpoints = ['/nfs/gatsbystor/ammarica/Supervised/Checkpoints_supervised_half_width_b_1024_lr_005',
'/nfs/gatsbystor/ammarica/Supervised/Checkpoints_supervised_b_1024_lr_05_new',
'/nfs/gatsbystor/ammarica/Supervised/Checkpoints_supervised_wide_b_1024_lr_05_new']

# define name and width
all_name = sorted(name*len(width))
all_width = width*len(name)
all_datasets = [datasets[1]]*len(width)*2
all_datasets = all_datasets + [datasets[0]]*len(width)*(len(name)-2)

# create all checkpoints based on name order - check before
all_checkpoints = ecoset_simclr_checkpoints + ecoset_sup_checkpoints + same_acc_sup_checkpoints + simclr_checkpoints + supervised_checkpoints 

# make dataframe and save
dict = {'name':all_name, 'width':all_width, 'dataset':all_datasets, 'checkpoint':all_checkpoints}
df = pd.DataFrame(dict)
df.to_csv('checkpoints_to_analyze.csv')


read_df = pd.read_csv('checkpoints_to_analyze.csv')
for i in range(read_df.shape[0]):
        this_pretrained = read_df.iloc[i]
        this_name = this_pretrained['name']
        this_width = this_pretrained['width']
        this_model = this_name + ' ' + str(this_width) + ' width'
        this_path = this_pretrained['checkpoint']
        this_checkpoint = f'{this_path}/model_best.pth.tar'
        print("=> loading checkpoint '{}'".format(this_model))
#print(read_df.iloc[0])