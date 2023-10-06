import cv2
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from torchvision import transforms
from collections import Counter
from sklearn.model_selection import KFold


class TrainDataset(Dataset):
    def __init__(self, path_list, bbox_list, labels, img_size):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.labels = labels
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        image = cv2.imread(self.path_list[index])

            
        image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
        try:
            image = cv2.resize(image, (self.img_size, self.img_size))
        except:
            print('Error')
            print(self.path_list[index])
            return -1
        
        i = np.random.randint(4)
        if i == 0:
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
            image = self.normalize(image)
            label = self.labels[index].type(torch.float32)

        elif i == 1:
            image = np.fliplr(image).copy()    
            image = torch.as_tensor(image).permute(2, 0, 1).float()
            image = self.normalize(image)
            
            label = self.labels[index].type(torch.float32)

        elif i == 2:
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = torch.as_tensor(image).permute(2, 0, 1).float()
            image = self.normalize(image)

            label = self.labels[index].type(torch.float32)

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
            
            label = self.labels[index].type(torch.float32)

        return image, label
        
    def __len__(self):
        return len(self.path_list)

class ValidDataset(Dataset):
    def __init__(self, path_list, bbox_list, label_list1, img_size):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.label_list1 = label_list1
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        image = cv2.imread(self.path_list[index])
        image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.as_tensor(image).permute(2, 0, 1).float()
        image = self.normalize(image)
        label = self.label_list1[index]
        
        return image, label

    def __len__(self):
        return len(self.path_list)
    
class ValidDataset_TTA(Dataset):
    def __init__(self, path_list, bbox_list, label_list1, img_size):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.label_list1 = label_list1
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
    def __getitem__(self, index):
        images = []
        for i in range(5):
            image = cv2.imread(self.path_list[index])
            image = image[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
            image = cv2.resize(image, (self.img_size, self.img_size))

            if i == 0:
                image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
                image = self.normalize(image)
                label = self.label_list1[index].type(torch.float32)

            elif i == 1:
                image = np.fliplr(image).copy()    
                image = torch.as_tensor(image).permute(2, 0, 1).float()
                image = self.normalize(image)
                label = self.label_list1[index].type(torch.float32)

            elif i == 2:
                image = cv2.GaussianBlur(image, (5, 5), 0)
                image = torch.as_tensor(image).permute(2, 0, 1).float()
                image = self.normalize(image)
                label = self.label_list1[index].type(torch.float32)

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
                label = self.label_list1[index].type(torch.float32)

            else:
                idx = np.random.randint(len(self.path_list))
                image1 = cv2.imread(self.path_list[index])
                image2 = cv2.imread(self.path_list[idx])
                image1 = image1[self.bbox_list[index][1]:self.bbox_list[index][3], self.bbox_list[index][0]:self.bbox_list[index][2]]
                image2 = image2[self.bbox_list[idx][1]:self.bbox_list[idx][3], self.bbox_list[idx][0]:self.bbox_list[idx][2]]
                image1 = cv2.resize(image1, (self.img_size, self.img_size))
                image2 = cv2.resize(image2, (self.img_size, self.img_size))
                image1 = self.normalize(image1)
                image2 = self.normalize(image2)
                image1 = torch.as_tensor(image1).permute(2, 0, 1).float()
                image2 = torch.as_tensor(image2).permute(2, 0, 1).float()

                label1 = self.label_list1[index].type(torch.float32)
                label1 = self.label_list1[idx].type(torch.float32)
                l = np.random.beta(0.2, 0.2)

                image = (l * image1) + ((1 - l) * image2)
                label = (l * label1) + ((1 - l) * label1)

                images.append(image)

        return images, label
        
    def __len__(self):
        return len(self.path_list)


def train_(CFG, category):
    df1 = pd.read_csv('./Dataset/info_etri20_emotion_train.csv')
    if os.path.exists(f'./Dataset/{category}_labels.csv'):
        print(f'{category} : Adding Additional Datas...')
        df2 = pd.read_csv(f'./Dataset/{category}_labels.csv')
        df = pd.concat([df1,df2[1:]])
    else:
        df = df1
    path = './Dataset/Train/'
    x = [path + i for i in df['image_name']]
    df['image_name'] = x
    
    path_train = df['image_name'].values
    bbox_train = df.iloc[:,1:5].values
    label_train = F.one_hot(torch.as_tensor(df[category].values))

    if 'Daily' in category:
        class_counts = Counter(df[category].values)
        min_count = min(class_counts.values())
        target_count = min_count * 300
        sampling_strategy = {cls: min(target_count, count) for cls, count in class_counts.items()}
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        indices_resampled, _ = rus.fit_resample(np.arange(len(path_train)).reshape(-1, 1), df[category].values)
        indices_resampled = indices_resampled.flatten()
        path_train_resampled = path_train[indices_resampled]
        bbox_train_resampled = bbox_train[indices_resampled]
        label_train_resampled = label_train[indices_resampled]
        train_dataset = TrainDataset(path_train_resampled, bbox_train_resampled, label_train_resampled, CFG['IMG_SIZE'])

    elif 'Gender' in category:
        class_counts = Counter(df[category].values)
        min_count = min(class_counts.values())
        target_count = min_count * 1
        sampling_strategy = {cls: min(target_count, count) for cls, count in class_counts.items()}
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        indices_resampled, _ = rus.fit_resample(np.arange(len(path_train)).reshape(-1, 1), df[category].values)
        indices_resampled = indices_resampled.flatten()
        path_train_resampled = path_train[indices_resampled]
        bbox_train_resampled = bbox_train[indices_resampled]
        label_train_resampled = label_train[indices_resampled]
        train_dataset = TrainDataset(path_train_resampled, bbox_train_resampled, label_train_resampled, CFG['IMG_SIZE'])

    if 'Embel' in category:
        class_counts = Counter(df[category].values)
        min_count = min(class_counts.values())
        target_count = min_count * 200
        sampling_strategy = {cls: min(target_count, count) for cls, count in class_counts.items()}
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        indices_resampled, _ = rus.fit_resample(np.arange(len(path_train)).reshape(-1, 1), df[category].values)
        indices_resampled = indices_resampled.flatten()
        path_train_resampled = path_train[indices_resampled]
        bbox_train_resampled = bbox_train[indices_resampled]
        label_train_resampled = label_train[indices_resampled]
        train_dataset = TrainDataset(path_train_resampled, bbox_train_resampled, label_train_resampled, CFG['IMG_SIZE'])
    
    else:
        train_dataset = TrainDataset(path_train, bbox_train, label_train, CFG['IMG_SIZE'])

    print(f'{category}: {len(label_train_resampled)}')
    weight = compute_class_weight('balanced', classes=np.unique(df[category].values), y=df[category].values)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=7)
    
    return weight, train_loader

def valid_(df2, path2 = './Dataset/Valid/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))

    valid_dataset1 = ValidDataset(path_valid, bbox, label_daily_valid, 224)
    valid_dataset2 = ValidDataset(path_valid, bbox, label_gender_valid, 224)
    valid_dataset3 = ValidDataset(path_valid, bbox, label_embel_valid, 224)
    
    return valid_dataset1, valid_dataset2, valid_dataset3

def valid_tta(df2, path2 = './Dataset/Valid/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))

    valid_dataset1 = ValidDataset_TTA(path_valid, bbox, label_daily_valid, 224)
    valid_dataset2 = ValidDataset_TTA(path_valid, bbox, label_gender_valid, 224)
    valid_dataset3 = ValidDataset_TTA(path_valid, bbox, label_embel_valid, 224)
    
    return valid_dataset1, valid_dataset2, valid_dataset3


def test_(df2, path2 = '/aif/Dataset/Test/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))

    valid_dataset1 = ValidDataset(path_valid, bbox, label_daily_valid, 224)
    valid_dataset2 = ValidDataset(path_valid, bbox, label_gender_valid, 224)
    valid_dataset3 = ValidDataset(path_valid, bbox, label_embel_valid, 224)
    
    return valid_dataset1, valid_dataset2, valid_dataset3

def test_tta(df2, path2 = '/aif/Dataset/Test/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = F.one_hot(torch.as_tensor(df2['Daily'].values))
    label_gender_valid = F.one_hot(torch.as_tensor(df2['Gender'].values))
    label_embel_valid = F.one_hot(torch.as_tensor(df2['Embellishment'].values))

    valid_dataset1 = ValidDataset_TTA(path_valid, bbox, label_daily_valid, 224)
    valid_dataset2 = ValidDataset_TTA(path_valid, bbox, label_gender_valid, 224)
    valid_dataset3 = ValidDataset_TTA(path_valid, bbox, label_embel_valid, 224)
    
    return valid_dataset1, valid_dataset2, valid_dataset3


def train_kfold(CFG, category):
    df1 = pd.read_csv('./Dataset/info_etri20_emotion_train.csv')
    if os.path.exists(f'./Dataset/{category}_labels.csv'):
        print(f'{category} : Adding Additional Datas...')
        df2 = pd.read_csv(f'./Dataset/{category}_labels.csv')
        df = pd.concat([df1,df2[1:]])
    else:
        df = df1
    path = './Dataset/Train/'
    x = [path + i for i in df['image_name']]
    df['image_name'] = x
    
    path_train = df['image_name'].values
    bbox_train = df.iloc[:,1:5].values
    label_train = F.one_hot(torch.as_tensor(df[category].values))
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds=[]
    for train_idx, valid_idx in kf.split(path_train):
        folds.append((train_idx, valid_idx))

    class_counts = Counter(df[category].values)
    min_count = min(class_counts.values())
    target_count = min_count * 20
    sampling_strategy = {cls: min(target_count, count) for cls, count in class_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    indices_resampled, _ = rus.fit_resample(np.arange(len(path_train)).reshape(-1, 1), df[category].values)
    indices_resampled = indices_resampled.flatten()
    path_train_resampled = path_train[indices_resampled]
    bbox_train_resampled = bbox_train[indices_resampled]
    label_train_resampled = label_train[indices_resampled]
    weight = compute_class_weight('balanced', classes=np.unique(df[category].values), y=df[category].values)

    train_dataset = TrainDataset(path_train_resampled, bbox_train_resampled, label_train_resampled, CFG['IMG_SIZE'])
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=15)
    
    return weight, train_loader

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
