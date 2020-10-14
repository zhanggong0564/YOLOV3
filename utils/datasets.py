import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
import os

def decode(image,label):
    pass
def encode(image,label):
    pass

def resize(image, size1,size2):
    image = F.interpolate(image.unsqueeze(0), size=(size1,size2), mode="nearest").squeeze(0)
    # print(image.shape)
    return image
class ListDateset(Dataset):
    def __init__(self,list_path,img_size = 416,transform = None):
        # with open(list_path,'r') as file:
        #     self.img_files = file.readlines()
        # self.img_files = glob.glob(os.path.join(list_path,'image/*'))
        with open(os.path.join(list_path,'train.txt'),'r') as f:
            self.img_files = f.readlines()
        # self.label_files = [path.replace('image', 'label').replace('.png', '.txt').replace('.jpg', '.txt') for path in
        #                     self.img_files]
        self.label_files = [os.path.join(list_path,'label', path.split('\\')[-1].replace('jpg','txt')) for path in
                            self.img_files]
        self.img_shape = (img_size,img_size)
        self.max_objects = 50
        self.transform = transform
    def __getitem__(self, index):
        img_path = self.img_files[index].rstrip()
        # print(img_path)
        # if img_path==r'E:\pytorch\utils\face_mask_data\train\test_00002232.jpg':
        #     print()
        # img = np.array(Image.open(img_path).convert('RGB'))
        img = cv2.imread(img_path)
        # print(img.shape)
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index].rstrip()
            img = np.array(Image.open(img_path))
        label_path = self.label_files[index].rstrip()
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)#c,x,y,w,h
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:  # 将更新后的box坐标填充到刚刚申请的占位空间中
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # image,labels = decode(img,filled_labels)
        # if self.transform:
        #     input_img,filled_labels= self.transform(image,labels)
        filled_labels = torch.from_numpy(filled_labels)
        input_img = np.transpose(img, (2,0,1))
        # 将图片转化成 tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = resize(input_img, *self.img_shape)
        # print(input_img.size())
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

