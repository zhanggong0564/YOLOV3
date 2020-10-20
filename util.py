from __future__ import division
import torch
import numpy as np
import math



def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    '''
    :param prediction:输出 n,c,h,w
    :param inp_dim:输入图像尺寸
    :param anchors:
    :param num_classes:类别
    :param CUDA:是否使用cuda
    :return:
    '''
    batch_size = prediction.size(0)
    stride = inp_dim//prediction.size(2)
    gride_size = prediction.size(2)
    bbox_attrs = 5+num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,gride_size*gride_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size,gride_size*gride_size*num_anchors,bbox_attrs)


    anchors = [(a[0]/stride, a[1]/stride) for  a in anchors]

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])


    grid = np.arange(gride_size)
    a,b = np.meshgrid(grid,grid)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(gride_size * gride_size, 1).unsqueeze(0)

    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction



def bbox_iou(box1, box2, x1y1x2y2=True):
    # 返回 box1 和 box2 的 iou, box1 和 box2 的 shape 要么相同, 要么其中一个为[1,4]
    if not x1y1x2y2:
        # 获取 box1 和 box2 的左上角和右下角坐标
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # 获取 box1 和 box2 的左上角和右下角坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 获取相交矩形的左上角和右下角坐标
    # 注意, torch.max 函数要求输入的两个参数要么 shape 相同, 此时在相同位置上进行比较并取最大值
    # 要么其中一个 shape 的第一维为 1, 此时会自动将该为元素与另一个 box 的所有元素做比较, 这里使用的就是该用法.
    # 具体来说, b1_x1 为 [1, 1], b2_x1 为 [3, 1], 此时会有 b1_x1 中的一条数据分别与 b2_x1 中的三条数据做比较并取最大值
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 计算相交矩形的面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # 分别求 box1 矩形和 box2 矩形的面积.
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # 计算 iou 并将其返回
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area
def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    # 参数:
    # pred_boxes: [1, 3, 13, 13, 4]
    # pred_conf: [1, 3, 13, 13]
    # pred_cls: [1, 3, 13, 13, 80]
    # target: [1, 50, 5]
    # anchors: [3, 2]
    # num_anchors: 3
    # num_classes: 80
    # grid_size: 13(特征图谱的尺寸)
    # ignore_thres: 0.5
    # img_dim: 图片尺寸

    nB = target.size(0) # batch_size
    nA = num_anchors # 3
    nC = num_classes # 80
    nG = grid_size # 特征图谱的尺寸(eg: 13)
    mask = torch.zeros(nB, nA, nG, nG) # eg: [1, 3, 13, 13], 代表每个特征图谱上的 anchors 下标(每个 location 都有 3 个 anchors)
    conf_mask = torch.ones(nB, nA, nG, nG) # eg: [1, 3, 13, 13] 代表每个 anchor 的置信度.
    tx = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的中心坐标
    ty = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的中心坐标
    tw = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的宽
    th = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的高
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0) # 占位空间, 存放置信度, eg: [1, 3, 13, 13]
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0) # 占位空间, 存放分类预测值, eg:[1, 3, 13, 13, 80]

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0: # b指定的batch中的某图片, t指定了图片中的某 box(按顺序)
                continue # 如果 box 的5个值(从标签到坐标)都为0, 那么就跳过当前的 box
            nGT += 1 # 每找到一个非零的 box, 则真实box的数量就加一

            # Convert to position relative to box
            # 由于我们在存储box的坐标时, 就是按照其相对于图片的宽和高的比例存储的
            # 因此, 当想要获取特征图谱上的对应 box 的坐标时, 直接令其与特征图谱的尺寸相乘即可.
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG

            # Get grid box indices
            # 获取在特征图谱上的整数坐标

            gi = int(gx)
            gj = int(gy)

            # Get shape of gt box, 根据 box 的大小获取 shape: [1,4]
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

            # Get shape of anchor box
            # 相似的方法得到anchor的shape: [3, 4] , 3 代表3个anchor
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))


            # 调用本文件的 bbox_iou 函数计算gt_box和anchors之间的交并比
            # 注意这里仅仅计算的是 shape 的交并比, 此处没有考虑位置关系.
            # gt_box 为 [1,4], anchors 为 [3, 4],
            # 最终返回的值为[3], 代表了 gt_box 与每个 anchor 的交并比大小
            anch_ious = bbox_iou(gt_box, anchor_shapes)

            # 将交并比大于阈值的部分设置conf_mask的对应位为0(ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # 找到匹配度最高的 anchor box, 返回下标: 0,1,2 中的一个
            best_n = np.argmax(anch_ious)

            # 获取相应的 ground truth box, unsqueeze用于扩充维度, 使[4]变成[1,4], 以便后面的计算
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # 获取最佳的预测 box, pred_boxes的shape为: [1,3,13,13,4]
            # pred_box经过unsqueeze扩充后的shape为: [1,4]
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)

            # 设置 mask 和 conf_mask
            mask[b, best_n, gj, gi] = 1
            # 注意, 刚刚将所有大于阈值的 conf_mask对应为都设置为了0,
            # 然后这里将具有最大交并比的anchor设置为1, 如此确保一个真实框只对应一个 anchor.
            # 由于 conf_mask 的默认值为1, 因此, 剩余的box可看做是负样本
            conf_mask[b, best_n, gj, gi] = 1

            # 设置中心坐标, 该坐标是相对于 cell的左上角而言的, 所以是一个小于1的数
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # 设置宽和高, 注意, 这里会转化成训练时使用的宽高值
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)


            # 获取当前 box 的 标签
            target_label = int(target[b, t, 0])
            # tcls: [1,3,13,13,80]
            # 将当前true box对应的 anchor 的正确类别设置为1
            tcls[b, best_n, gj, gi, target_label] = 1
            # 将置信度设置为 1
            tconf[b, best_n, gj, gi] = 1

            # 调用 bbox_iou 函数计算 ground truth 和最佳匹配的预测box之间的 iou
            # 注意, 此时的 gt_box为 [gx,gy,gw,gh], 不是 [tx,ty,tw,th]
            # gt_box的shape为[1,4], pred_box为最佳匹配的预测 box, 其shape也为[1,4]
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            # pred_cls的shape为[1,3,13,13,80], 获取最佳匹配anchor box的最大概率类别的下标
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            # pred_conf的shape为[1,3,13,13], 获取最佳匹配anchor box的置信度
            score = pred_conf[b, best_n, gj, gi]
            # if iou > 0.5 and pred_label == target_label and score > 0.5:
            #     nCorrect += 1 # 如果 iou 和 score 大于阈值, 并且标签预测正确, 则正确项增1

    # 将所有需要的信息都返回, 从这里可以看出, 每一个 YOLO 层都会执行一次预测.
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls