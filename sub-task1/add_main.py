import os
import numpy as np

import torch

# from utils.dataset_underSample import train_, valid_, test_
from utils.dataset_add import train_, valid_tta, valid_
# from utils.dataset_randAug import train_, test_
# from utils.dataset_underSample import train_, valid_, test_
from add_res_net import Resnet, ResNetResidualConnectionModel, ResNetFeedForwardModel
from add_coatNet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4
from rexnet import Rexnet
from utils.utils import CosineAnnealingWarmUpRestarts, Lion
from add_train import train
import pandas as pd
import torch.utils.data

CFG = {
    'IMG_SIZE':224,
    'PATCH_SIZE':4,
    'PROJECTION_DIM':96,
    'SHIFT_PIXEL':1,
    'NUM_DIV':12,
    'DROP_RATE':0.5,
    'DEPTHS':[6,8,18,6],   #[6,8,18,6]
    'EPOCHS':300,
    'INIT_LR':0,
    'MAX_LR':1e-4,
    'BATCH_SIZE':32,
    'SEED':14,
    'ES_CNT':20,
    'STEP':1
}

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])

def main():
    if os.path.exists('models') is False:
        os.makedirs('models')
    df2 = pd.read_csv('./Dataset/info_etri20_emotion_validation.csv')
    w1, train_loader1 = train_(CFG, 'Daily')
    w2, train_loader2 = train_(CFG, 'Gender')
    w3, train_loader3 = train_(CFG, 'Embellishment')
    valid_dataset1, valid_dataset2, valid_dataset3 = valid_(df2)

    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
    valid_loader1 = torch.utils.data.DataLoader(valid_dataset1, batch_size=16, shuffle=False, num_workers=7)
    valid_loader2 = torch.utils.data.DataLoader(valid_dataset2, batch_size=16, shuffle=False, num_workers=7)
    valid_loader3 = torch.utils.data.DataLoader(valid_dataset3, batch_size=16, shuffle=False, num_workers=7)
    model = ResNetResidualConnectionModel().to(device)
    # model = coatnet_2()
    # trained_weights = torch.load('./models/mixup/task1_681', map_location=device) # 자기 모델 경로를 지정합니다
    # model.load_state_dict(trained_weights)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG['INIT_LR'],weight_decay=1e-1)
    # optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG['INIT_LR'])
    # optimizer = Lion(params=model.parameters(), lr=CFG['INIT_LR'])
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=CFG['MAX_LR'], gamma=0.7)

    train(model, optimizer, train_loader1, train_loader2, train_loader3, w1, w2, w3, valid_loader1, valid_loader2, valid_loader3, scheduler, CFG, device)

if __name__ == '__main__':
    main()