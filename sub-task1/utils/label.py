def main():
    from dataset_randAug import _test
    from res_net import Resnet

    import pandas as pd
    import numpy as np

    import torch
    import torch.utils.data
    import torch.utils.data.distributed

    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Resnet().to(device)
    net.load_state_dict(torch.load('./models/best/val80', map_location=device))

    df = pd.read_csv('./Dataset/sexy.csv') 
    val_dataset = _test(df, './Dataset/sexy') 
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0) 

    daily_pred_list = np.array([])
    gender_pred_list = np.array([])
    embel_pred_list = np.array([])

    with torch.no_grad():
        for images, d_labels, g_labels, e_labels in tqdm(iter(val_dataloader)):
            images, d_labels, g_labels, e_labels = images.to(device), d_labels.to(device), g_labels.to(device), e_labels.to(device)
            
            out_daily, out_gender, out_embel = net(images)

            _, daily_indx = out_daily.max(1)
            daily_pred_list = np.concatenate([daily_pred_list, daily_indx.cpu()], axis=0)

            _, gender_indx = out_gender.max(1)
            gender_pred_list = np.concatenate([gender_pred_list, gender_indx.cpu()], axis=0)

            _, embel_indx = out_embel.max(1)
            embel_pred_list = np.concatenate([embel_pred_list, embel_indx.cpu()], axis=0)

    out = pd.DataFrame({'image_name':df['image_name'], 'daily':daily_pred_list, 'gender':gender_pred_list, 'embel':embel_pred_list})
    out.to_csv('./Dataset/sexy_pred.csv', index=False)
   
    return out

if __name__ == '__main__':
    main()