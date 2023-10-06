# from utils.dataset_randAug import _test
# from utils.dataset_add import *
from utils.dataset_one import *
from add_coatNet import coatnet_0, coatnet_1, coatnet_2, coatnet_3, coatnet_4

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.utils.data
import torch.utils.data.distributed
# from add_res_net import Resnet, ResNetResidualConnectionModel
from res_net import Resnet, ResNetResidualConnectionModel
from tqdm import tqdm
from utils.utils import *

def inf_one(model, val_dataloader, device):
    model.eval()
    daily_gt_list = np.array([])
    daily_pred_list = np.array([])
    gender_gt_list = np.array([])
    gender_pred_list = np.array([])
    embel_gt_list = np.array([])
    embel_pred_list = np.array([])

    criterion = DiceLoss().to(device)

    total_daily_loss = []
    total_gender_loss = []
    total_embel_loss = []
    with torch.no_grad():
        for images, d_labels, g_labels, e_labels in tqdm(iter(val_dataloader)):
            images, d_labels, g_labels, e_labels = images.to(device), d_labels.to(device), g_labels.to(device), e_labels.to(device)
            d_outputs, g_outputs, e_outputs = model(images)

            daily_indx = torch.argmax(d_outputs, dim=1)
            gender_indx = torch.argmax(g_outputs, dim=1)
            embel_indx = torch.argmax(e_outputs, dim=1)

            daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)
            gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)
            embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)

            daily_gt = np.array(d_labels.cpu())
            gender_gt = np.array(g_labels.cpu())
            embel_gt = np.array(e_labels.cpu())

            daily_idx = np.array([np.argmax(daily_gt,axis=1)]).squeeze()
            gender_idx = np.array([np.argmax(gender_gt,axis=1)]).squeeze()
            embel_idx = np.array([np.argmax(embel_gt,axis=1)]).squeeze()

            daily_gt_list = np.concatenate([daily_gt_list, daily_idx], axis=0)
            gender_gt_list = np.concatenate([gender_gt_list, gender_idx], axis=0)
            embel_gt_list = np.concatenate([embel_gt_list, embel_idx], axis=0)

            daily_loss = criterion(d_outputs.clone().detach().to(device), torch.from_numpy(daily_gt).to(device))
            gender_loss = criterion(g_outputs.clone().detach().to(device), torch.from_numpy(gender_gt).to(device))
            embel_loss = criterion(e_outputs.clone().detach().to(device), torch.from_numpy(embel_gt).to(device))

            total_daily_loss.append(daily_loss.item())
            total_gender_loss.append(gender_loss.item())
            total_embel_loss.append(embel_loss.item())

    total_daily_loss = np.mean(total_daily_loss)
    total_gender_loss = np.mean(total_gender_loss)
    total_embel_loss = np.mean(total_embel_loss)
    total_loss_avg = (total_daily_loss + total_gender_loss + total_embel_loss) / 3
    daily_top_1, daily_acsa, d_a, d_b, d_c, d_d = get_test_metrics(daily_gt_list, daily_pred_list)
    gender_top_1, gender_acsa, g_a, g_b, g_c, g_d = get_test_metrics(gender_gt_list, gender_pred_list)
    embel_top_1, embel_acsa, e_a, e_b, e_c, e_d = get_test_metrics(embel_gt_list, embel_pred_list)
    print("------------------------TRAIN LOSS-------------------------")
    print(
        "Daily:(Top-1=%.5f, ACSA=%.5f), Gender:(Top-1=%.5f, ACSA=%.5f), Embellishment:(Top-1=%.5f, ACSA=%.5f)" % (
            daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa))
    
    print("------------------------VAL LOSS---------------------------")
    print(f"VAL Loss: {total_loss_avg:.6f}, daily_loss: {total_daily_loss:.6f}, gender_loss: {total_gender_loss:.6f}, embel_loss: {total_embel_loss}")
    out = (daily_top_1 + gender_top_1 + embel_top_1) / 3
    acsa = (daily_acsa + gender_acsa + embel_acsa ) / 3
    print('Total Validation Accuracy: ', out)
    print('Total Validation ACSA: ', acsa)
    print('$$$ ', (d_a + g_a + e_a) / 3)
    print('$$$ ', (d_b + g_b + e_b) / 3)
    print('$$$ ', (d_c + g_c + e_c) / 3)
    print('$$$ ', (d_d + g_d + e_d) / 3)
    print("-----------------------------------------------------------")    

    return out, acsa, total_loss_avg
@torch.no_grad()
def inf_three(model, val_dataloader1, val_dataloader2, val_dataloader3, device):
    model.eval()
    daily_gt_list = np.array([])
    daily_pred_list = np.array([])
    gender_gt_list = np.array([])
    gender_pred_list = np.array([])
    embel_gt_list = np.array([])
    embel_pred_list = np.array([])

    criterion = DiceLoss().to(device)
    total_daily_loss = []
    total_gender_loss = []
    total_embel_loss = []
    with torch.no_grad():
        for batch1, batch2, batch3 in tqdm(zip(val_dataloader1, val_dataloader2, val_dataloader3)):
            images1, labels1 = batch1
            images2, labels2 = batch2
            images3, labels3 = batch3
            
            images1 = images1.to(device)
            images2 = images2.to(device)
            images3 = images3.to(device)
            
            d_labels = labels1.to(device)
            g_labels = labels2.to(device)
            e_labels = labels3.to(device)

            d_outputs = model(images1,'d')
            g_outputs = model(images2,'g')
            e_outputs = model(images3,'e')
        
            daily_gt = np.array(d_labels.cpu())
            daily_idx = np.array([np.argmax(daily_gt,axis=1)]).squeeze()
            daily_gt_list = np.concatenate([daily_gt_list, daily_idx], axis=0)

            gender_gt = np.array(g_labels.cpu())
            gender_idx = np.array([np.argmax(gender_gt,axis=1)]).squeeze()
            gender_gt_list = np.concatenate([gender_gt_list, gender_idx], axis=0)
            
            embel_gt = np.array(e_labels.cpu())
            embel_idx = np.array([np.argmax(embel_gt,axis=1)]).squeeze()
            embel_gt_list = np.concatenate([embel_gt_list, embel_idx], axis=0)

            daily_pred = d_outputs
            daily_indx = torch.argmax(daily_pred, dim=1)
            daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)

            gender_pred = g_outputs
            gender_indx = torch.argmax(gender_pred, dim=1)
            gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)

            embel_pred = e_outputs
            embel_indx = torch.argmax(embel_pred, dim=1)
            embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)


            # Validation loss 추가
            daily_loss = criterion(d_outputs.clone().detach().to(device), torch.from_numpy(daily_gt).to(device))
            total_daily_loss.append(daily_loss.item())

            gender_loss = criterion(g_outputs.clone().detach().to(device), torch.from_numpy(gender_gt).to(device))
            total_gender_loss.append(gender_loss.item())

            embel_loss = criterion(e_outputs.clone().detach().to(device), torch.from_numpy(embel_gt).to(device))
            total_embel_loss.append(embel_loss.item())

    total_daily_loss = np.mean(total_daily_loss)
    total_gender_loss = np.mean(total_gender_loss)
    total_embel_loss = np.mean(total_embel_loss)
    total_loss_avg = (total_daily_loss + total_gender_loss + total_embel_loss) / 3

    daily_top_1, daily_acsa, d_a, d_b, d_c, d_d = get_test_metrics(daily_gt_list, daily_pred_list)
    gender_top_1, gender_acsa, g_a, g_b, g_c, g_d = get_test_metrics(gender_gt_list, gender_pred_list)
    embel_top_1, embel_acsa, e_a, e_b, e_c, e_d = get_test_metrics(embel_gt_list, embel_pred_list)
    print("------------------------TRAIN LOSS-------------------------")
    print(
        "Daily:(Top-1=%.5f, ACSA=%.5f), Gender:(Top-1=%.5f, ACSA=%.5f), Embellishment:(Top-1=%.5f, ACSA=%.5f)" % (
            daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa))
    
    print("------------------------VAL LOSS---------------------------")
    print(f"VAL Loss: {total_loss_avg:.6f}, daily_loss: {total_daily_loss:.6f}, gender_loss: {total_gender_loss:.6f}, embel_loss: {total_embel_loss}")
    out = (daily_top_1 + gender_top_1 + embel_top_1) / 3
    acsa = (daily_acsa + gender_acsa + embel_acsa ) / 3
    print('Total Validation Accuracy: ', out)
    print('Total Validation ACSA: ', acsa)
    print('$$$ ', (d_a + g_a + e_a) / 3)
    print('$$$ ', (d_b + g_b + e_b) / 3)
    print('$$$ ', (d_c + g_c + e_c) / 3)
    print('$$$ ', (d_d + g_d + e_d) / 3)
    print("-----------------------------------------------------------")    

    return out, acsa, total_loss_avg

def inf_tta(model, val_dataloader1, val_dataloader2, val_dataloader3, device):
    model.eval()
    daily_gt_list = np.array([])
    daily_pred_list = np.array([])
    gender_gt_list = np.array([])
    gender_pred_list = np.array([])
    embel_gt_list = np.array([])
    embel_pred_list = np.array([])

    criterion = DiceLoss().to(device)
    total_daily_loss = []
    total_gender_loss = []
    total_embel_loss = []
    with torch.no_grad():
        for batch1, batch2, batch3 in tqdm(zip(val_dataloader1, val_dataloader2, val_dataloader3)):
            d_output = []
            g_output = []
            e_output = []
            images1, labels1 = batch1
            images2, labels2 = batch2
            images3, labels3 = batch3

            d_labels = labels1.to(device)
            g_labels = labels2.to(device)
            e_labels = labels3.to(device)

            for img in images1:
                img = img.to(device)
                d_output.append(model(img,'d'))
            
            for img in images2:
                img = img.to(device)
                g_output.append(model(img,'g'))
            
            for img in images3:
                img = img.to(device)
                e_output.append(model(img,'e'))

            d_outputs = sum(d_output) / len(d_output)
            g_outputs = sum(g_output) / len(g_output)
            e_outputs = sum(e_output) / len(e_output)

            daily_indx = torch.argmax(d_outputs, dim=1)
            gender_indx = torch.argmax(g_outputs, dim=1)
            embel_indx = torch.argmax(e_outputs, dim=1)

            daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)
            gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)
            embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)

            daily_gt = np.array(d_labels.cpu())
            gender_gt = np.array(g_labels.cpu())
            embel_gt = np.array(e_labels.cpu())

            daily_idx = np.array([np.argmax(daily_gt,axis=1)]).squeeze()
            gender_idx = np.array([np.argmax(gender_gt,axis=1)]).squeeze()
            embel_idx = np.array([np.argmax(embel_gt,axis=1)]).squeeze()
            
            daily_gt_list = np.concatenate([daily_gt_list, daily_idx], axis=0)
            gender_gt_list = np.concatenate([gender_gt_list, gender_idx], axis=0)
            embel_gt_list = np.concatenate([embel_gt_list, embel_idx], axis=0)

            daily_loss = criterion(d_outputs.clone().detach().to(device), torch.from_numpy(daily_gt).to(device))
            gender_loss = criterion(g_outputs.clone().detach().to(device), torch.from_numpy(gender_gt).to(device))
            embel_loss = criterion(e_outputs.clone().detach().to(device), torch.from_numpy(embel_gt).to(device))

            total_daily_loss.append(daily_loss.item())
            total_gender_loss.append(gender_loss.item())
            total_embel_loss.append(embel_loss.item())

    total_daily_loss = np.mean(total_daily_loss)
    total_gender_loss = np.mean(total_gender_loss)
    total_embel_loss = np.mean(total_embel_loss)

    total_loss_avg = (total_daily_loss + total_gender_loss + total_embel_loss) / 3

    daily_top_1, daily_acsa, d_a, d_b, d_c, d_d = get_test_metrics(daily_gt_list, daily_pred_list)
    gender_top_1, gender_acsa, g_a, g_b, g_c, g_d = get_test_metrics(gender_gt_list, gender_pred_list)
    embel_top_1, embel_acsa, e_a, e_b, e_c, e_d = get_test_metrics(embel_gt_list, embel_pred_list)
    print("------------------------TRAIN LOSS-------------------------")
    print(
        "Daily:(Top-1=%.5f, ACSA=%.5f), Gender:(Top-1=%.5f, ACSA=%.5f), Embellishment:(Top-1=%.5f, ACSA=%.5f)" % (
            daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa))
    
    print("------------------------VAL LOSS---------------------------")
    print(f"VAL Loss: {total_loss_avg:.6f}, daily_loss: {total_daily_loss:.6f}, gender_loss: {total_gender_loss:.6f}, embel_loss: {total_embel_loss}")
    out = (daily_top_1 + gender_top_1 + embel_top_1) / 3
    acsa = (daily_acsa + gender_acsa + embel_acsa ) / 3
    print('Total Validation Accuracy: ', out)
    print('Total Validation ACSA: ', acsa)
    print('$$$ ', (d_a + g_a + e_a) / 3)
    print('$$$ ', (d_b + g_b + e_b) / 3)
    print('$$$ ', (d_c + g_c + e_c) / 3)
    print('$$$ ', (d_d + g_d + e_d) / 3)
    print("-----------------------------------------------------------")    

    return out, acsa, total_loss_avg

def one_tta(model, val_dataloader, device):
    model.eval()
    daily_gt_list = np.array([])
    daily_pred_list = np.array([])
    gender_gt_list = np.array([])
    gender_pred_list = np.array([])
    embel_gt_list = np.array([])
    embel_pred_list = np.array([])

    criterion = DiceLoss().to(device)
    total_daily_loss = []
    total_gender_loss = []
    total_embel_loss = []
    with torch.no_grad():
        for images, d_labels, g_labels, e_labels in tqdm(iter(val_dataloader)):
            d_labels, g_labels, e_labels = d_labels.to(device), g_labels.to(device), e_labels.to(device)

            d_output = []
            g_output = []
            e_output = []

            for img in images:
                img = img.to(device)
                d_out, g_out, e_out = model(img)
                d_output.append(d_out)
                g_output.append(g_out)
                e_output.append(e_out)

            d_outputs = sum(d_output) / len(d_output)
            g_outputs = sum(g_output) / len(g_output)
            e_outputs = sum(e_output) / len(e_output)

            daily_indx = torch.argmax(d_outputs, dim=1)
            gender_indx = torch.argmax(g_outputs, dim=1)
            embel_indx = torch.argmax(e_outputs, dim=1)

            daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)
            gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)
            embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)

            daily_gt = np.array(d_labels.cpu())
            gender_gt = np.array(g_labels.cpu())
            embel_gt = np.array(e_labels.cpu())

            daily_idx = np.array([np.argmax(daily_gt,axis=1)]).squeeze()
            gender_idx = np.array([np.argmax(gender_gt,axis=1)]).squeeze()
            embel_idx = np.array([np.argmax(embel_gt,axis=1)]).squeeze()
            
            daily_gt_list = np.concatenate([daily_gt_list, daily_idx], axis=0)
            gender_gt_list = np.concatenate([gender_gt_list, gender_idx], axis=0)
            embel_gt_list = np.concatenate([embel_gt_list, embel_idx], axis=0)

            daily_loss = criterion(d_outputs.clone().detach().to(device), torch.from_numpy(daily_gt).to(device))
            gender_loss = criterion(g_outputs.clone().detach().to(device), torch.from_numpy(gender_gt).to(device))
            embel_loss = criterion(e_outputs.clone().detach().to(device), torch.from_numpy(embel_gt).to(device))

            total_daily_loss.append(daily_loss.item())
            total_gender_loss.append(gender_loss.item())
            total_embel_loss.append(embel_loss.item())

    total_daily_loss = np.mean(total_daily_loss)
    total_gender_loss = np.mean(total_gender_loss)
    total_embel_loss = np.mean(total_embel_loss)

    total_loss_avg = (total_daily_loss + total_gender_loss + total_embel_loss) / 3

    daily_top_1, daily_acsa, d_a, d_b, d_c, d_d = get_test_metrics(daily_gt_list, daily_pred_list)
    gender_top_1, gender_acsa, g_a, g_b, g_c, g_d = get_test_metrics(gender_gt_list, gender_pred_list)
    embel_top_1, embel_acsa, e_a, e_b, e_c, e_d = get_test_metrics(embel_gt_list, embel_pred_list)
    print("------------------------TRAIN LOSS-------------------------")
    print(
        "Daily:(Top-1=%.5f, ACSA=%.5f), Gender:(Top-1=%.5f, ACSA=%.5f), Embellishment:(Top-1=%.5f, ACSA=%.5f)" % (
            daily_top_1, daily_acsa, gender_top_1, gender_acsa, embel_top_1, embel_acsa))
    
    print("------------------------VAL LOSS---------------------------")
    print(f"VAL Loss: {total_loss_avg:.6f}, daily_loss: {total_daily_loss:.6f}, gender_loss: {total_gender_loss:.6f}, embel_loss: {total_embel_loss}")
    out = (daily_top_1 + gender_top_1 + embel_top_1) / 3
    acsa = (daily_acsa + gender_acsa + embel_acsa ) / 3
    print('Total Validation Accuracy: ', out)
    print('Total Validation ACSA: ', acsa)
    print('$$$ ', (d_a + g_a + e_a) / 3)
    print('$$$ ', (d_b + g_b + e_b) / 3)
    print('$$$ ', (d_c + g_c + e_c) / 3)
    print('$$$ ', (d_d + g_d + e_d) / 3)
    print("-----------------------------------------------------------")    

    return out, acsa, total_loss_avg

def get_test_metrics(y_true, y_pred, verbose=False):
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)

    cnf_matrix = confusion_matrix(y_true, y_pred)
    if verbose:
        print(cnf_matrix)

    TP = np.diag(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - TP
    FN = cnf_matrix.sum(axis=1) - TP
    TN = cnf_matrix.sum() - (FP + FN + TP)

    top_1 = np.sum(TP)/np.sum(np.sum(cnf_matrix))
    cs_accuracy = TP / (FN + TP)
    spec = TN / (TN + FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    negative = TN / (TN + FN)
    precision = TP / (TP + FP)

    return top_1, cs_accuracy.mean(), spec.mean(), acc.mean(), negative.mean(), precision.mean()

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    df2 = pd.read_csv('./Dataset/info_etri20_emotion_validation.csv')
    # valid_dataset1, valid_dataset2, valid_dataset3 = (df2)
    # valid_loader1 = torch.utils.data.DataLoader(valid_dataset1, batch_size=4, shuffle=False, num_workers=5)
    # valid_loader2 = torch.utils.data.DataLoader(valid_dataset2, batch_size=4, shuffle=False, num_workers=5)
    # valid_loader3 = torch.utils.data.DataLoader(valid_dataset3, batch_size=4, shuffle=False, num_workers=5)
    valid_dataset = valid_tta(df2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=7)


    model = ResNetResidualConnectionModel().to(device)
    model.load_state_dict(torch.load(f"/home/dcvlab/dcv/Coco/models/submitted/1_loss_61_487"))

    # _, _val_loss = inf_tta(model, valid_loader1, valid_loader2, valid_loader3, device)

    _, _, _ = one_tta(model, valid_loader, device)
