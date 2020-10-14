import torch.nn as nn
from util import *


class YOLOLoss(nn.Module):

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors # anchors = [(116,90),(156,198),(373,326)] ,anchors = [(30,61),(62,45),(59,119)],anchors = [(10,13),(16,30),(33,23)]
        self.num_anchors = len(anchors) # 3
        self.num_classes = num_classes # 80
        self.bbox_attrs = 5 + num_classes #
        self.image_dim = img_dim # 416
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        # x: [1, 255, 13, 13]
        # targets: [50, 5]
        nA = self.num_anchors # 3
        nB = x.size(0) # batch_size
        nG = x.size(2) # W = 13
        stride = self.image_dim / nG # 416 / W = 416 / 13 = 32

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.BoolTensor if x.is_cuda else torch.ByteTensor
        # (batch, anchors, 5+num_classes, x.size(2), x.size(2)), 调换顺序
        # [1, 3, 13, 13, 85]
        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0,1,3,4,2).contiguous()

        x = torch.sigmoid(prediction[..., 0]) # center x: [1, 3, 13, 13]
        y = torch.sigmoid(prediction[..., 1]) # center y: [1, 3, 13, 13]
        w = prediction[..., 2] # width: [1, 3, 13, 13]
        h = prediction[..., 3] # height: [1, 3, 13, 13]
        pred_conf = torch.sigmoid(prediction[..., 4]) # [1, 3, 13, 13]
        pred_cls = torch.sigmoid(prediction[..., 5:]) # [1, 3, 13, 13, 80]

        # grid_x的shape为[1,1,nG,nG], 每一行的元素为:[0,1,2,3,...,nG-1]
        grid_x = torch.arange(nG).repeat(nG, 1).view([1,1,nG,nG]).type(FloatTensor)
        # grid_y的shape为[1,1,nG,nG], 每一列元素为: [0,1,2,3, ...,nG-1]
        grid_y = torch.arange(nG).repeat(nG, 1).t().view(1,1,nG,nG).type(FloatTensor)

        # scaled_anchors 是将原图上的 box 大小根据当前特征图谱的大小转换成相应的特征图谱上的 box
        # shape: [3, 2]
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])

        # 分别获取其 w 和 h, 并将shape形状变为: [1,3,1,1]
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # shape: [1, 3, 13, 13, 4], 给 anchors 添加 offset 和 scale
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        #bx by bw bh
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is not None:# 如果提供了 targets 标签, 则说明是处于训练阶段

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            # 调用 utils.py 文件中的 build_targets 函数, 将真实的 box 数据转化成训练用的数据格式
            # nGT = int 真实box的数量
            # nCorrect = int 预测正确的数量
            # mask: torch.Size([1, 3, 13, 13])
            # conf_mask: torch.Size([1, 3, 13, 13])
            # tx: torch.Size([1, 3, 13, 13])
            # ty: torch.Size([1, 3, 13, 13])
            # tw: torch.Size([1, 3, 13, 13])
            # th: torch.Size([1, 3, 13, 13])
            # tconf: torch.Size([1, 3, 13, 13])
            # tcls: torch.Size([1, 3, 13, 13, 80])
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, cls = build_targets(
                pred_boxes = pred_boxes.cpu().data,
                pred_conf=pred_cls.cpu().data,
                pred_cls = pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item()) # 计算置信度大于0.5的预测box数量
            recall = float(nCorrect / nGT) if nGT else 1 # 计算召回率,预测正确的个数/总真实的个数
            # precision = float(nCorrect / nProposals)

            # 处理 masks
            mask = mask.type(ByteTensor)
            conf_mask = conf_mask.type(ByteTensor)

            # 处理 target Variables
            tx = tx.type(FloatTensor)
            ty = ty.type(FloatTensor)
            tw = tw.type(FloatTensor)
            th = th.type(FloatTensor)
            tconf = tconf.type(FloatTensor)
            tcls = cls.type(LongTensor)

            # 获取表明gt和非gt的conf_mask
            # 这里 conf_mask_true 指的是具有最佳匹配度的anchor box
            # conf_mask_false 指的是iou小于0.5的anchor box, 其余的anchor box都被忽略了
            conf_mask_true = mask # mask 只有best_n对应位为1, 其余都为0
            conf_mask_false = (conf_mask.int()-mask.int()).bool() # conf_mask中iou大于ignore_thres的为0, 其余为1, best_n也为1

            # 忽略 non-existing objects, 计算相应的loss mse均方差loos
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])

            # 这里 conf_mask_true 指的是具有最佳匹配度的anchor box
            # conf_mask_false 指的是iou小于0.5的anchor box, 其余的anchor box都被忽略了
            #bceloss就是不带sigmoid,二分类
            loss_conf = self.bce_loss(
                pred_conf[conf_mask_false], tconf[conf_mask_false]
            ) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )

            # pred_cls[mask]的shape为: [7,80], torch.argmax(tcls[mask], 1)的shape为[7]
            # CrossEntropyLoss对象的输入为(x,class), 其中x为预测的每个类的概率, class为gt的类别下标
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))

            loss = 10*(loss_x + loss_y + loss_w + loss_h) + loss_conf + loss_cls
            #
            # return (
            #     loss,
            #     loss_x.item(),
            #     loss_y.item(),
            #     loss_w.item(),
            #     loss_h.item(),
            #     loss_conf.item(),
            #     loss_cls.item(),
            #     recall,
            #     precision,
            # )
            return loss
        # else:
        #     # 非训练阶段则直接返回准确率, output的shape为: [nB, -1, 85]
        #     output = torch.cat(
        #         (
        #             pred_boxes.view(nB, -1, 4) * stride,
        #             pred_conf.view(nB, -1, 1),
        #             pred_cls.view(nB, -1, self.num_classes),
        #         ),
        #         -1,
        #     )
        #
        #     return output

