from utils.utils import *
from tqdm import tqdm
import os
import torch
import numpy as np
import torch.nn as nn
from test import inf_three, inf_tta
from itertools import zip_longest

scaler = torch.cuda.amp.GradScaler()


def train(model, optimizer, train_loader1, train_loader2, train_loader3, w1, w2, w3, valid_loader1, valid_loader2, valid_loader3, scheduler, CFG, device):
    
    if not os.path.exists("./models/mixup/"):
        os.mkdir("./models/mixup/")
    
    model.to(device)
    w1 = torch.FloatTensor(w1).to(device)
    w2 = torch.FloatTensor(w2).to(device)
    w3 = torch.FloatTensor(w3).to(device)
    # d_criterion = nn.CrossEntropyLoss().to(device)
    # g_criterion = SoftLabelSmoothingLoss(classes=5, smoothing=0.1).to(device)
    # e_criterion = nn.CrossEntropyLoss().to(device)

    # accumulation_steps = CFG['STEP']
    best_val_loss = 999
    patience_check = 0

    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        
        for i, (batch1, batch2, batch3) in enumerate(tqdm(zip_longest(train_loader1, train_loader2, train_loader3))):
            loss_parts = []

            if batch1 is not None:
                images1, labels1 = batch1
                images1 = images1.to(device)
                d_labels = labels1.to(device)
                d_outputs = model(images1, 'd')
                d_loss1 = DiceLoss()(d_outputs, d_labels).to(device)
                d_loss2 = FocalLoss(w1)(d_outputs, d_labels).to(device)
                loss_parts.extend([d_loss1, d_loss2])

            if batch2 is not None:
                images2, labels2 = batch2
                images2 = images2.to(device)
                g_labels = labels2.to(device)
                g_outputs = model(images2, 'g')
                g_loss1 = DiceLoss()(g_outputs, g_labels).to(device)
                g_loss2 = FocalLoss(w2)(g_outputs, g_labels).to(device)
                loss_parts.extend([g_loss1, g_loss2])

            if batch3 is not None:
                images3, labels3 = batch3
                images3 = images3.to(device)
                e_labels = labels3.to(device)
                e_outputs = model(images3, 'e')
                e_loss1 = DiceLoss()(e_outputs, e_labels).to(device)
                e_loss2 = FocalLoss(w3)(e_outputs, e_labels).to(device)
                loss_parts.extend([e_loss1, e_loss2])

            if loss_parts:
                loss = sum(loss_parts) / len(loss_parts) / CFG['STEP']
                train_loss.append(loss.item())
                scaler.scale(loss).backward()

            if (i % CFG['STEP'] == 0) or (i == len(train_loader2)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        print(f'\nEpoch [{epoch}]  Learning Rate [{scheduler.get_lr()[0]}]\n')
        # out, _val_loss = inf_tta(model, valid_loader1, valid_loader2, valid_loader3, device)
        out, acsa, loss = inf_three(model, valid_loader1, valid_loader2, valid_loader3, device)

        if scheduler is not None:
            scheduler.step()
        
        if best_val_loss > loss:
            best_val_loss = loss
            torch.save(model.state_dict(), f"./models/mixup/add_best")
            patience_check = 0
            print('***** Best Model *****')

        else:
            patience_check += 1
            print('Early Stopping Count: ', patience_check)
            if patience_check % CFG['ES_CNT'] == 0:
                CFG['ES_CNT'] = CFG['ES_CNT'] // 2
                print('Loading the best model...')
                trained_weights = torch.load('./models/mixup/add_best', map_location=device)
                model.load_state_dict(trained_weights)
            
            if CFG['ES_CNT'] == 5:
                break