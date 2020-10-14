import torch
import numpy as np
import math

def bbox_iou(gt_box,anchor_s,is_cycx=False):
    if not is_cycx:
        b1_x1,b1_y1,b1_x2,b1_y2 = gt_box[:,0],gt_box[:,1],gt_box[:,2],gt_box[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = anchor_s[:, 0], anchor_s[:, 1], anchor_s[:, 2], anchor_s[:, 3]

        left_x = torch.max(b1_x1,b2_x1)
        left_y = torch.max(b1_y1,b2_y1)
        right_x = torch.min(b1_x2,b2_x2)
        right_y = torch.min(b1_y2,b2_y2)

        inter_area = torch.clamp(right_x-left_x+1,min=0)*torch.clamp(right_y-left_y+1,min=0)
        b1_area = (b1_x2-b1_x1+1)*(b1_y2-b1_y1+1)
        b2_area = (b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)
        iou = inter_area/(b1_area+b2_area-inter_area+1e-16)
    else:
        gx,gy,gw,gh = gt_box
        px,py,pw,ph = anchor_s
        b1_x1, b1_y1, b1_x2, b1_y2 = gx-(gw/2),gy-(gh/2),gx+(gw/2),gy+(gh/2)
        b2_x1, b2_y1, b2_x2, b2_y2 = px - (pw / 2), py - (ph / 2), px + (pw / 2), py + (ph / 2)

        left_x = torch.max(b1_x1, b2_x1)
        left_y = torch.max(b1_y1, b2_y1)
        right_x = torch.min(b1_x2, b2_x2)
        right_y = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(right_x - left_x + 1, min=0) * torch.clamp(right_y - left_y + 1, min=0)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def  build_target(target,anchors,num_classes,pred_boxes,pred_clf,pred_conf,ignore_thres):
    nb = target.size(0)
    na = len(anchors)
    nc = num_classes
    ng = pred_boxes.size(3)

    #1,3,13,13,c
    mask = torch.zeros(nb,na,ng,ng)
    noob_mask = torch.ones(nb,na,ng,ng)

    tx = torch.zeros(nb,na,ng,ng)
    ty = torch.zeros(nb,na,ng,ng)
    tw = torch.zeros(nb,na,ng,ng)
    th = torch.zeros(nb,na,ng,ng)

    tconf = torch.zeros(nb,na,ng,ng)
    tcls = torch.zeros(nb,na,ng,ng,nc)

    nGT = 0
    nCorrect = 0
    for b in range(nb):
        for t in range(target.shape(1)):
            if target[b,t].sum() == 0:
                continue
            nGT+=1
            gx = target[b,t,1]*ng
            gy = target[b,t,2]*ng
            gw = target[b,t,3]*ng
            gh = target[b,t,3]*ng

            gi = int(gx)
            gj = int(gy)

            gt_box = torch.from_numpy(np.array(
                [0,0,gw,gh]
            )).unsqueeze(0)
            anchor_s = torch.FloatTensor(np.concatenate([np.zeros((na,2)),np.array(anchors)],1))

            anchor_iou = bbox_iou(gt_box,anchor_s)
            noob_mask[b,anchor_iou>ignore_thres,gi,gj] = 0
            best_iou = np.argmax(anchor_iou)
            mask[b,best_iou,gj, gi] = 1

            tx[b,best_iou,gj, gi] = gx-gi
            ty[b, best_iou, gj, gi] = gy - gj
            tw[b,best_iou,gj, gi] = math.log(gw/anchors[best_iou][0]+1e-16)
            th[b, best_iou, gj, gi] = math.log(gh / anchors[best_iou][1] + 1e-16)

            target_label = int(target[b,t,0])
            tcls[b,best_iou,gj, gi,target_label] = 1
            tconf[b,best_iou,gj, gi] = 1
            gt_bbox =[gx,gy,gw,gh]
            pred_boxe = pred_boxes[b, best_iou, gj, gi]
            iou = bbox_iou(gt_bbox,pred_boxe)
            pred_label = torch.argmax(pred_clf[b,best_iou,gj, gi])
            score = pred_conf[b,best_iou,gj, gi]
            if iou>0.5 and pred_label==target_label and score>0.5:
                nCorrect+=1
            return mask,noob_mask,tx,ty,tw,th,tconf,tcls,nCorrect