from utils.utils import *
from tqdm import tqdm
import torch.nn
import os
import torch
import torch.nn as nn
import numpy as np
from test import inf_three, inf_tta, inf_one
import wandb

scaler = torch.cuda.amp.GradScaler()

def train(model, optimizer, train_loader, weights, val_loader, scheduler, CFG, device):

    wandb.watch(model, log='all')

    if not os.path.exists("./models/mixup/"):
        os.mkdir("./models/mixup/")
    
    model.to(device)
    w1 = torch.FloatTensor(weights[0]).to(device)
    w2 = torch.FloatTensor(weights[1]).to(device)
    w3 = torch.FloatTensor(weights[2]).to(device)
    d_loss1, g_loss1, e_loss1 = build_f_loss('focal1', w1, w2, w3)
    loss2 = build_ce_loss('dice')

    best_val_loss = 999
    best_asca = -999
    patience_check = 0
    
    for epoch in range(1, CFG['EPOCHS']+1):
        lr = scheduler.get_lr()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr[0],weight_decay=0.083)
        optimizer.zero_grad()
        model.train()
        train_loss = []
        
        for i, (images, d_labels, g_labels, e_labels) in enumerate(tqdm(iter(train_loader))):
            images, d_labels, g_labels, e_labels = images.to(device), d_labels.to(device), g_labels.to(device), e_labels.to(device)
            
            d_outputs, g_outputs, e_outputs = model(images)
            # d_loss = DiceLoss()(d_outputs, d_labels) + FocalLoss(w1)(d_outputs, d_labels)
            # g_loss = DiceLoss()(g_outputs, g_labels) + FocalLoss(w2)(g_outputs, g_labels)
            # e_loss = DiceLoss()(e_outputs, e_labels) + FocalLoss(w3)(e_outputs, e_labels)

            if d_loss1 != None:
                d_loss = d_loss1(d_outputs, d_labels) + loss2(d_outputs, d_labels)
                g_loss = g_loss1(g_outputs, g_labels) + loss2(g_outputs, g_labels)
                e_loss = e_loss1(e_outputs, e_labels) + loss2(e_outputs, e_labels)

            else:
                d_loss = loss2(d_outputs, d_labels)*2
                g_loss = loss2(g_outputs, g_labels)*2
                e_loss = loss2(e_outputs, e_labels)*2

            d_loss = d_loss.to(device)
            e_loss = e_loss.to(device)
            g_loss = g_loss.to(device)

            loss = (d_loss + g_loss + e_loss) / 6
            train_loss.append(loss.item())
            scaler.scale(loss).backward()

            if (i % CFG['STEP'] == 0) or (i == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
        print(f'Epoch [{epoch}]  Learning Rate [{scheduler.get_lr()[0]}]')
        
        out, acsa, loss = inf_one(model, val_loader, device)

        wandb.log({'d_loss': d_loss, 'e_loss': e_loss, 'g_loss': g_loss, 'loss': loss, 'ascc': out, 'acsa': acsa}, step=epoch)
        
        if scheduler is not None:
            scheduler.step()
        
        if best_val_loss > loss + 0.002:
            best_val_loss = loss
            torch.save(model.state_dict(), f"./models/mixup/loss_best")
            patience_check = 0
            print('***** Best Loss Model *****')

        else:
            patience_check += 1
            print('Early Stopping Count: ', patience_check)
            if patience_check % CFG['ES_CNT'] == 0:
                CFG['ES_CNT'] = CFG['ES_CNT'] // 3
                patience_check = 0
                if CFG['ES_CNT'] < 3:
                    break
                trained_weights = torch.load('./models/mixup/model_best', map_location=device)
                model.load_state_dict(trained_weights)

        if best_asca < acsa:
            best_asca = acsa
            torch.save(model.state_dict(), f"./models/mixup/acsa_best")
            print('***** Best ACSA Model *****')

        wandb.run.summary['best_val_loss'] = best_val_loss

def build_f_loss(l, w1, w2, w3):
    if l == "focal1":
        _d_loss = FocalLoss(w1)
        _g_loss = FocalLoss(w2)
        _e_loss = FocalLoss(w3)
    elif l == "focal2":
        _d_loss = FocalLoss1(w1)
        _g_loss = FocalLoss1(w2)
        _e_loss = FocalLoss1(w3)
    elif l == "focalt":
        _d_loss = FocalTverskyLoss()
        _g_loss = FocalTverskyLoss()
        _e_loss = FocalTverskyLoss()
    elif l == "none":
        _d_loss = None
        _g_loss = None
        _e_loss = None

    return _d_loss, _g_loss, _e_loss

def build_ce_loss(l):
    if l == "jaccard":
        loss = JaccardLoss()
    elif l == "dice":
        loss = DiceLoss()
    elif l == "ce":
        loss = nn.CrossEntropyLoss()

    return loss

def build_optimizer(network, optimizer, lr, weight_decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=lr, momentum=0.9)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(network.parameters(),
                               lr=lr, weight_decay=weight_decay)
    elif optimizer == "Lion":
        optimizer = Lion(params=network.parameters(), lr=lr, weight_decay=weight_decay)  
    return optimizer