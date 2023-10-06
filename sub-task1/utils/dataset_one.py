import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, path_list, bbox_list, d_labels, g_labels, e_labels, img_size):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.d_labels = d_labels
        self.g_labels = g_labels
        self.e_labels = e_labels
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        image = cv2.imread(self.path_list[index])
        image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        i = np.random.randint(4)
        if i == 0:
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
            image = self.normalize(image)

        elif i == 1:
            image = np.fliplr(image).copy()    
            image = torch.as_tensor(image).permute(2, 0, 1).float()
            image = self.normalize(image)

        elif i == 2:
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = torch.as_tensor(image).permute(2, 0, 1).float()
            image = self.normalize(image)

        elif i == 3:
            h, w, c= image.shape
            gauss = np.random.randn(h, w, c)
            sigma = 12.5
            noise = gauss * sigma
            image = image + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = torch.as_tensor(image).permute(2, 0, 1).float()
            image = self.normalize(image)

        d_label = self.d_labels[index].type(torch.float32)
        g_label = self.g_labels[index].type(torch.float32)
        e_label = self.e_labels[index].type(torch.float32)

        return image, d_label, g_label, e_label
        
    def __len__(self):
        return len(self.path_list)

class ValidDataset(Dataset):
    def __init__(self, path_list, bbox_list, d_labels, e_labels, g_labels, img_size):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.d_labels = d_labels
        self.g_labels = e_labels
        self.e_labels = g_labels
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        image = cv2.imread(self.path_list[index])
        image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.as_tensor(image).permute(2, 0, 1).float()
        image = self.normalize(image)
        d_label = self.d_labels[index].type(torch.float32)
        g_label = self.g_labels[index].type(torch.float32)
        e_label = self.e_labels[index].type(torch.float32)
        
        return image, d_label, g_label, e_label

    def __len__(self):
        return len(self.path_list)
    
class ValidDataset_TTA(Dataset):
    def __init__(self, path_list, bbox_list, d_labels, e_labels, g_labels, img_size):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.d_label = d_labels
        self.g_label = e_labels
        self.e_label = g_labels
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        images = []
        for i in range(4):
            image = cv2.imread(self.path_list[index])
            image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
            image = cv2.resize(image, (self.img_size, self.img_size))

            if i == 0:
                image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
                image = self.normalize(image)

            elif i == 1:
                image = np.fliplr(image).copy()    
                image = torch.as_tensor(image).permute(2, 0, 1).float()
                image = self.normalize(image)

            elif i == 2:
                image = cv2.GaussianBlur(image, (5, 5), 0)
                image = torch.as_tensor(image).permute(2, 0, 1).float()
                image = self.normalize(image)

            elif i == 3:
                h, w, c= image.shape
                gauss = np.random.randn(h, w, c)
                sigma = 12.5
                noise = gauss * sigma
                image = image + noise
                image[image > 255] = 255
                image[image < 0] = 0
                image = torch.as_tensor(image).permute(2, 0, 1).float()
                image = self.normalize(image)
            
            d_label = self.d_label[index].type(torch.float32)
            g_label = self.g_label[index].type(torch.float32)
            e_label = self.e_label[index].type(torch.float32)
            images.append(image)

        return images, d_label, g_label, e_label
        
    def __len__(self):
        return len(self.path_list)


def train_(CFG):
    weights = []
    df = pd.read_csv('./Dataset/info_etri20_emotion_train.csv')
    path = './Dataset/Train/'
    x = [path + i for i in df['image_name']]
    df['image_name'] = x
    
    path_train = df['image_name'].values
    bbox_train = df.iloc[:,1:5].values
    label_daily_train = F.one_hot(torch.as_tensor(df['Daily'].values))
    label_gender_train = F.one_hot(torch.as_tensor(df['Gender'].values))
    label_embel_train = F.one_hot(torch.as_tensor(df['Embellishment'].values))
    train_dataset = TrainDataset(path_train, bbox_train, label_daily_train, label_gender_train, label_embel_train, CFG['IMG_SIZE'])

    weights.append(compute_class_weight('balanced', classes=np.unique(df['Daily'].values), y=df['Daily'].values))
    weights.append(compute_class_weight('balanced', classes=np.unique(df['Gender'].values), y=df['Gender'].values))
    weights.append(compute_class_weight('balanced', classes=np.unique(df['Embellishment'].values), y=df['Embellishment'].values))
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=7)
    
    return weights, train_loader

def valid_(df2, path2 = './Dataset/Valid/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))
    valid_dataset = ValidDataset(path_valid, bbox, label_daily_valid, label_gender_valid, label_embel_valid, 224)

    return valid_dataset

def valid_tta(df2, path2 = './Dataset/Valid/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))
    valid_dataset = ValidDataset_TTA(path_valid, bbox, label_daily_valid, label_gender_valid, label_embel_valid, 224)

    return valid_dataset

def test_(df2, path2 = '/aif/Dataset/Test/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))
    valid_dataset = ValidDataset(path_valid, bbox, label_daily_valid, label_gender_valid, label_embel_valid, 224)

    return valid_dataset

def test_tta(df2, path2 = '/aif/Dataset/Test/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))
    valid_dataset = ValidDataset_TTA(path_valid, bbox, label_daily_valid, label_gender_valid, label_embel_valid, 224)

    return valid_dataset

def pad_to_square(image):
    h, w, c = image.shape
    diff = abs(h-w)
    top, bottom, left, right = 0, 0, 0, 0
    
    if h < w:
        top = diff // 2
        bottom = diff - top
    elif w < h:
        left = diff // 2
        right = diff - left
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(1,1,1))
    return image
