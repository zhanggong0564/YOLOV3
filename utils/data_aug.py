import torch
import random
import numpy as np
import cv2

def show(img,labels):
    img = img.copy()
    for lab in labels:
        if np.sum(lab)!=0:
            cv2.rectangle(img,(int(lab[1]),int(lab[2])),(int(lab[3]),int(lab[4])),(0,255,255))
            cv2.imshow('image',img)
            cv2.waitKey()

class RandomHorizontalFilp(object):
    def __init__(self,p = 0.5):
        self.p = p
    def __call__(self,sampler):
        # print("翻转")
        img, labels = sampler['image'],sampler['label']
        _,w,_=img.shape
        img = img[:,::-1,:]
        labels[:,[1,3]] = w-labels[:,[1,3]]
        sampler = {'image':img,'label':labels}
        return sampler


class RandomCrop(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sampler):
        img, labels = sampler['image'], sampler['label']

        if random.random()<self.p:
            # show(img, labels)
            # print("随机裁剪")
            h,w,_ = img.shape
            # max_bbox = np.concatenate(
            #     [
            #         np.min(labels[:,1],axis=0),
            #         np.min(labels[:,2], axis=0),
            #         np.max(labels[:,3],axis=0),
            #         np.max(labels[:,4], axis=0)
            #     ],
            #     axis=-1
            # )
            x1 =np.min(labels[:,3],axis=0)
            y1 =np.min(labels[:,2], axis=0)
            x2 = np.max(labels[:,1],axis=0)
            y2 = np.max(labels[:,4], axis=0)

            max_l_trans = x1
            max_u_trans = y1
            max_r_trans = w-x2
            max_d_trans = h-y2

            crop_xmin = max(0, int(x1 - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(y1- random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(x2 + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(y2 + random.uniform(0, max_d_trans)))

            img = img[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
            labels[:,[1,3]] = labels[:,[1,3]] -crop_xmin
            labels[:,[2,4]] = labels[:,[2,4]] -crop_ymin
            # show(img,labels)
            sampler = {'image':img,'label':labels}
            return sampler
        else:
            sampler = {'image': img, 'label': labels}
            return sampler

class RandomAffine(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sampler):
        img, labels = sampler['image'], sampler['label']
        if random.random()<self.p:
            # show(img, labels)
            # print("左右平移")
            h,w,_ = img.shape
            # max_bbox = np.concatenate(
            #     [
            #         np.min(labels[:,1:3],axis=0),
            #         np.max(labels[:,3:5],axis=0)
            #     ],
            #     axis=-1
            # )
            x1 = np.min(labels[:, 3], axis=0)
            y1 = np.min(labels[:, 2], axis=0)
            x2 = np.max(labels[:, 1], axis=0)
            y2 = np.max(labels[:, 4], axis=0)
            max_l_trans = x1
            max_u_trans = y1
            max_r_trans = w-x2
            max_d_trans = h-y2

            tx = random.uniform(-(max_l_trans-1),(max_r_trans-1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([
                [1,0,tx],
                [0,1,ty]
            ])
            img = cv2.warpAffine(img,M,(w,h))
            labels[:, [1,3]] = labels[:, [1,3]] + tx
            labels[:, [2,4]] = labels[:, [2,4]] + ty
            # show(img, labels)
        sampler = {'image':img,'label':labels}
        return sampler

class Resize(object):
    def __init__(self,target_shape,correct_box = True):
        self.h_target,self.w_target = target_shape
        self.corret_box = correct_box
    def __call__(self,sampler):
        img, labels = sampler['image'], sampler['label']
        # show(img, labels)
        h_org,w_org,_ = img.shape
        resize_ratio = min(1.0*self.w_target/w_org,1.0*self.h_target/h_org)
        resize_w = int(resize_ratio*w_org)
        resize_h = int(resize_ratio*h_org)
        img_resize = cv2.resize(img,(resize_w,resize_h))
        image_pad = np.full((self.h_target,self.w_target,3),128)
        dw = (self.w_target-resize_w)//2
        dh = (self.h_target-resize_h)//2
        image_pad[dh:resize_h+dh,dw:resize_w+dw,:] = img_resize
        image = image_pad/255.0

        if self.corret_box:
            labels[:, [1,3]] = labels[:, [1,3]] * resize_ratio + dw
            labels[:, [2,4]] = labels[:, [2,4]] * resize_ratio + dh
            # show(image, labels)
            sampler = {'image': image, 'label': labels}
            return sampler

        return image
class LabelSmooth(object):
    def __init__(self,delta = 0.01):
        self.delta = delta
    def __call__(self,onehot,num_class):
        laebl = onehot*(1-self.delta)+self.delta*1/num_class
        return laebl

class ToTensor(object):
    def __call__(self, sampler):
        img, labels = sampler['image'], sampler['label']
        image = np.transpose(img,(2,0,1))
        image = image.astype(np.float32)
        return torch.from_numpy(image.copy()),torch.from_numpy(labels.copy())