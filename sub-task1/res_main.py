import os
import numpy as np
import torch

from utils.utils import *
import pandas as pd
import torch.utils.data

# from utils.dataset_randAug import _train, _test
# from utils.dataset_original import _train, _test
from utils.dataset_one import train_, valid_, valid_tta
from res_net import Resnet, ResNetResidualConnectionModel, ResNetFeedForwardModel
from coat_net import coatnet_0
from res_train import train
from rexnet import Rexnet
import wandb
import yaml

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':300,
    'INIT_LR':0,
    'MAX_LR':1e-4,
    'BATCH_SIZE':8,
    'SEED':14,
    'ES_CNT':30,
    'STEP':2,
    'D_loss_w': 1.0,
    'G_loss_w': 1.0,
    'E_loss_w': 1.0
}
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])

wandb.init(
    project='ETRI',
    notes='ResNet',
    tags=['undersample'],
    config={
     'INIT_LR':0,
     'MAX_LR':1e-4,
     'EPOCHS':300,
    }
)

# sweep_configuration = {
#     'method': 'random',
#     'name': 'sweep',
#     'metric':{'goal': 'minimize', 'name': 'loss'},
#     'parameters': 
#     {
#         'eta_max': {'max': 1e-4, 'min': 7e-5},
#         'gamma': {'max': 0.6, 'min': 0.4},
#         'weight_decay': {'max':6e-1, 'min': 1e-1},
#         'lr':{'max': 5e-5, 'min': 1e-5},
#         'optimizer': {'values': ["AdamW", "Lion"]},
#         'model': {'values': ["Resnet", "ResNetResidualConnectionModel", "ResNetFeedForwardModel"]},
#         'g_loss_w': {'max': 1.5, 'min': 0.7}
#     }
# }

def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "Lion":
        optimizer = Lion(params=network.parameters(), lr=learning_rate, weight_decay=weight_decay)  
    return optimizer

def build_model(name):
    if name == "Resnet":
        model = Resnet().to(device)
    elif name == "ResNetResidualConnectionModel":
        model = ResNetResidualConnectionModel().to(device)
    elif name == "ResNetFeedForwardModel":
        model = ResNetFeedForwardModel().to(device) 
    return model

# sweep_id = wandb.sweep(sweep=sweep_configuration, project='ETRI')

def main():
    run = wandb.init()

    if os.path.exists('models') is False:
        os.makedirs('models')

    df2 = pd.read_csv('/home/dcvlab/dcv/Coco/Dataset/info_etri20_emotion_validation.csv')
    weights, train_loader = train_(CFG)
    valid_dataset = valid_(df2)

    # lr = wandb.config.lr
    # batch_size = wandb.config.batch_size
    # weight_decay = wandb.config.weight_decay
    # eta_max = wandb.config.eta_max
    # gamma = wandb.config.gamma
    # g_loss_w = wandb.config.g_loss_w
    
    # CFG['G_loss_w'] = g_loss_w
    

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=7)
    # model = build_model(wandb.config.model).to(device)
    # optimizer = build_optimizer(model, wandb.config.optimizer, learning_rate=lr, weight_decay = weight_decay)
    # optimizer = Lion(params=model.parameters(), lr=lr, weight_decay= weight_decay)

    # model = ResNetResidualConnectionModel().to(device)
    model = ResNetFeedForwardModel().to(device)
    # model.load_state_dict(torch.load(f"/home/dcvlab/dcv/Coco/models/submitted/1_loss_61_487"))

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-5, weight_decay=0.1)


    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=7e-6, weight_decay = 0.4534)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=CFG['MAX_LR'], gamma=0.7)
    train(model, optimizer, train_loader, weights, valid_loader, scheduler, CFG, device)

# wandb.agent(sweep_id, function=main, count=30)


if __name__ == '__main__':
    main()