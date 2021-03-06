import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.data_aug import *
import matplotlib.pyplot as plt
from visdom import Visdom


def decode(image,label):
    _,h,w, = image.shape
    labels = np.copy(label)
    ##测试标签是否正确
    # image1 = image.numpy()*255
    # image1 = np.transpose(image1,(1,2,0)).astype(np.uint8).copy()
    #
    # for lab in labels:
    #     if np.sum(lab)!=0:
    #         cv2.rectangle(image1,(int(lab[1]),int(lab[2])),(int(lab[3]),int(lab[4])),(0,255,255))
    #         cv2.imshow('image',image1)
    #         cv2.waitKey()
    labels[:,1:3] =torch.abs(label[:,1:3]+label[:,3:5])/2
    labels[:,3:5] = torch.abs(label[:,3:5]-label[:,1:3])
    # print(labels)
    # labs = np.copy(labels)
    # labs[:, 1:3] = labels[:, 1:3] - labels[:,3: 5] / 2
    # labs[:, 3:5] = labels[:, 1:3] + labels[:,3: 5] / 2
    # for lab in labs:
    #     if np.sum(lab)!=0:
    #         cv2.rectangle(image1, (int(lab[1]), int(lab[2])), (int(lab[3]), int(lab[4])), (0, 255, 255))
    #         cv2.imshow('image',image1)
    #         cv2.waitKey()

    temp = np.array([1, w, h, w, h]).astype(np.float64)
    temp = torch.from_numpy(temp.copy())
    label = labels/temp
    return image,label
def encode(image,label):
    h, w, _ = image.shape
    temp = np.array([1,w,h,w,h])
    label = temp*label
    labels = np.copy(label)
    labels[:, 1:3] = label[:, 1:3] - label[:,3: 5] / 2
    labels[:, 3:5] = label[:, 1:3] + label[:,3: 5] / 2
    # for lab in labels:
    #     if np.sum(lab)!=0:
    #         cv2.rectangle(image,(int(lab[1]),int(lab[2])),(int(lab[3]),int(lab[4])),(0,255,255))
    #         cv2.imshow('image',image)
    #         cv2.waitKey()
    return image,labels

# def resize(image, size1,size2):
#     image = F.interpolate(image.unsqueeze(0), size=(size1,size2), mode="nearest").squeeze(0)
#     # print(image.shape)
#     return image
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
        img = cv2.imread(img_path)
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index].rstrip()
            img = np.array(Image.open(img_path))
        label_path = self.label_files[index].rstrip()
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)#c,x,y,w,h

        image,labels = encode(img,labels)
        sampler = {'image':image,'label':labels}
        if self.transform:
            input_img,labels= self.transform(sampler)
            image, label = decode(input_img,labels)
            for lab in label:
                if lab[1]>1:
                    print(img_path)

            filled_labels = torch.zeros((self.max_objects, 5),dtype=torch.float64)
            if label is not None:  # 将更新后的box坐标填充到刚刚申请的占位空间中
                filled_labels[range(len(label))[:self.max_objects]] = label[:self.max_objects]
            return image, filled_labels
    def __len__(self):
        return len(self.img_files)
def collate_fn(batch):
    label = []
    imgs = []
    for info in batch:
        imgs.append(info[0])
        label.append(info[1])
    return torch.stack(imgs,0),torch.stack(label,0)
if __name__ == '__main__':
    train_path = '../data'
    transform = transforms.Compose([RandomHorizontalFilp(), RandomCrop(), RandomAffine(), Resize((416, 416)), ToTensor()])
    dataset = ListDateset(train_path, transform=transform)
    dataloder = DataLoader(dataset, batch_size=1, collate_fn=collate_fn,shuffle=True)
    for epoch in range(0, 100):
        totol_loss = 0
        num = 0
        for i, (image, target) in enumerate(dataloder):
            image = image.squeeze(0)
            # image1 = image.numpy()*255
            # image1 = np.transpose(image1,(1,2,0)).astype(np.uint8).copy()
            # target = target.squeeze(0)
            # for lab in target:
            #     lab = lab.numpy()*[1,416,416,416,416]
            #     if np.sum(lab)!=0:
            #         cv2.circle(image1,(int(lab[1]),int(lab[2])),3,(0,0,255))
            #         cv2.imshow('image',image1)
            #         cv2.waitKey()
