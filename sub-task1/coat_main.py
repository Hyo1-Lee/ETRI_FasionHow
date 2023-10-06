import os
import numpy as np
import torch

from utils.utils import CosineAnnealingWarmUpRestarts
import pandas as pd
import torch.utils.data

# from utils.dataset_randAug import _train, _test
from utils.dataset_original import _train, _test
from res_net import Resnet
from coat_net import coatnet_0
from res_train import train

CFG = {
    'IMG_SIZE':224,
    'PATCH_SIZE':4,
    'PROJECTION_DIM':96,
    'SHIFT_PIXEL':1,
    'NUM_DIV':12,
    'DROP_RATE':0.5,
    'DEPTHS':[6,8,18,6],   #[6,8,18,6]
    'EPOCHS':300,
    'INIT_LR':1e-6,
    'MAX_LR':1e-4,
    'BATCH_SIZE':64,
    'SEED':14,
    'ES_CNT':50,
    'STEP_ACC':4
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])

def main():
    if os.path.exists('models') is False:
        os.makedirs('models')
    df2 = pd.read_csv('./Dataset/info_etri20_emotion_validation.csv')
    train_loader1 = _train(CFG)
    valid_dataset1= _test(df2)

    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
    valid_loader1 = torch.utils.data.DataLoader(valid_dataset1, batch_size=64, shuffle=False, num_workers=24)
    model = coatnet_0()
    model.eval()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG['INIT_LR'], weight_decay = 1e-3)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=CFG['MAX_LR'], T_up=5, gamma=0.9)
    train(model, optimizer, train_loader1,valid_loader1, scheduler, CFG, device)


if __name__ == '__main__':
    main()