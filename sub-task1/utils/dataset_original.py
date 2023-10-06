import torch.utils.data
import numpy as np
from torchvision import transforms
from skimage import io, transform, color
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class BackGround(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, landmarks, sub_landmarks=None):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')

        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]

            new_image = np.zeros((self.output_size, self.output_size, 3))

            if h > w:
                new_image[:,(112 - new_w//2):(112 - new_w//2 + new_w),:] = img
                landmarks = landmarks + [112 - new_w//2, 0]
            else:
                new_image[(112 - new_h//2):(112 - new_h//2 + new_h), :, :] = img
                landmarks = landmarks + [0, 112 - new_h//2]

            if sub_landmarks is not None:
                sub_landmarks = sub_landmarks * [new_w / w, new_h / h]
                if h > w:
                    sub_landmarks = sub_landmarks + [112 - new_w // 2, 0]
                else:
                    sub_landmarks = sub_landmarks + [0, 112 - new_h // 2]
                return new_image, landmarks, sub_landmarks
            else:
                return new_image, landmarks
        else:
            new_image = np.zeros((self.output_size, self.output_size, 3))
            if h > w:
                new_image[:,(112 - new_w//2):(112 - new_w//2 + new_w),:] = img
            else:
                new_image[(112 - new_h//2):(112 - new_h//2 + new_h), :, :] = img

            return new_image


class BBoxCrop(object):
    def __call__(self, image, x_1, y_1, x_2, y_2):
        h, w = image.shape[:2]

        top = y_1
        left = x_1
        new_h = y_2 - y_1
        new_w = x_2 - x_1

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class DatasetOriginal(torch.utils.data.Dataset):
    def __init__(self, path_list, bbox_list, daily_label, gender_label, embel_label, img_size):
        self.path_list = path_list
        self.bbox_list = bbox_list
        self.daily_label = daily_label
        self.gender_label = gender_label
        self.embel_label = embel_label
        self.img_size = img_size
        self.bbox_crop = BBoxCrop()
        self.background = BackGround(224)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.to_pil = transforms.ToPILImage()


    def __getitem__(self, i):
        sample = self.path_list[i]
        image = io.imread(sample)
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)

        image = self.bbox_crop(image, self.bbox_list[i][0], self.bbox_list[i][1], self.bbox_list[i][2], self.bbox_list[i][3])
        image = self.background(image, None)

        image_ = image.copy()

        image_ = self.to_tensor(image_)
        image_ = self.normalize(image_)
        image_ = image_.float()

        if "Val" not in self.path_list[i]:
            d_label = self.daily_label[i].type(torch.float32)
            g_label = self.gender_label[i].type(torch.float32)
            e_label = self.embel_label[i].type(torch.float32)
        else:
            d_label = self.daily_label[i]
            g_label = self.gender_label[i]
            e_label = self.embel_label[i]

        return image_, d_label, g_label, e_label

    def __len__(self):
        return len(self.path_list)

def _train(CFG):
    df1 = pd.read_csv('./Dataset/info_etri20_emotion_train.csv')

    path1 = './Dataset/Train/'
    df1['image_name'] = path1 + df1['image_name']

    path_train = df1['image_name'].values
    bbox_train = df1.iloc[:, 1:5].values
    label_daily_train = F.one_hot(torch.as_tensor(df1['Daily'].values))
    label_gender_train = F.one_hot(torch.as_tensor(df1['Gender'].values))
    label_embel_train = F.one_hot(torch.as_tensor(df1['Embellishment'].values))

    train_dataset = DatasetOriginal(path_train, bbox_train, label_daily_train, label_gender_train, label_embel_train, CFG['IMG_SIZE'])
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=15)

    return train_loader

def _test(df2, path2 = './Dataset/Valid/'):
    df2['image_name'] = path2 + df2['image_name']
    path_valid = df2['image_name'].values
    bbox = df2.iloc[:, 1:5].values
    label_daily_valid = torch.as_tensor(df2['Daily'].values)
    label_gender_valid = torch.as_tensor(df2['Gender'].values)
    label_embel_valid = torch.as_tensor(df2['Embellishment'].values)
    val_dataset = DatasetOriginal(path_valid, bbox, label_daily_valid, label_gender_valid, label_embel_valid, 224)
    return val_dataset