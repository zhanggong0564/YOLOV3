import torch
import random
import numpy as np
import cv2



class RandomHorizontalFilp(object):
    def __init__(self,p = 0.5):
        self.p = p
    def __call__(self,img,labels):
        _,w,_=img.shape
        img = img[:,::-1,:]
        labels[:,[0]] = w-labels[:,[0]]
        return img,labels


class RandomCrop(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img,labels):
        if random.random()<self.p:
            h,w,_ = img.shape
            max_bbox = np.concatenate(
                [
                    np.min(labels[:,:2],axis=0),
                    np.min(labels[:,2:4],axis=0)
                ],
                axis=-1
            )
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w-max_bbox[2]
            max_d_trans = h-max_bbox[3]

            crop_xmin = max(0,int(max_bbox[0]-random.uniform(0,max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin:crop_ymax,crop_xmin:crop_xmax]
            labels[:,[0,2]] = labels[:,[0,2]] -crop_xmin
            labels[:,[1,3]] = labels[:,[1,3]] -crop_ymin
        return img,labels

class RandomAffine(object):
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,img,labels):
        if random.random()<self.p:
            h,w,_ = img.shape
            max_bbox = np.concatenate(
                [
                    np.min(labels[:,:2],axis=-1),
                    np.min(labels[:,2:4],axis=-1)
                ],
                axis=-1
            )
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w-max_bbox[2]
            max_d_trans = h-max_bbox[3]

            tx = random.uniform(-(max_l_trans-1),(max_r_trans-1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([
                [1,0,tx],
                [0,1,ty]
            ])
            img = cv2.warpAffine(img,M,(w,h))
            labels[:,[0,2]] = labels[:,[0,2]] +tx
            labels[:, [1, 3]] = labels[:, [1, 3]] + ty
        return img,labels

class Resize(object):
    def __init__(self,target_shape,correct_box = True):
        self.h_target,self.w_target = target_shape
        self.corret_box = correct_box
    def __call__(self,img,labels):
        h_org,w_org,_ = img.shape
        resize_ratio = min(1.0*self.w_target/w_org,1.0*self.h_target/h_org)
        resize_w = int(resize_ratio*w_org)
        resize_h = int(resize_ratio*h_org)
        img_resize = cv2.resize(img,(resize_w,resize_h))
        image_pad = np.full(self.h_target,self.w_target)
        dw = (self.w_target-resize_w)//2
        dh = (self.h_target-resize_h)//2
        image_pad[dh:resize_h+dh,dw:resize_w+dw] = img_resize
        image = image_pad/255.0

        if self.corret_box:
            labels[:,[0,2]]  = labels[:,[0,2]]*resize_ratio +dw
            labels[:, [1, 3]] = labels[:, [1, 3]] * resize_ratio + dh
            return image,labels
        return image
class LabelSmooth(object):
    def __init__(self,delta = 0.01):
        self.delta = delta
    def __call__(self,onehot,num_class):
        laebl = onehot*(1-self.delta)+self.delta*1/num_class
        return laebl