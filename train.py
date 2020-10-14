# from utils.yololoss import YOLOLoss
from utils.datasets import *
from util import *
import torch
from net.yolov4 import YoloBody
from torch.utils.data import DataLoader
from yolov3 import YOLOLoss
from torchvision.models import resnet
import tqdm


train_path = './data'
anchors = [
    [(116,90),(156,198),(373,326)],
    [(30,61),(62,45),(59,119)],
    [(10,13),(16,30),(33,23)]
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YoloBody(3,4).to(device)
yololoss1 = YOLOLoss(anchors[0],4,416)
yololoss2 = YOLOLoss(anchors[1],4,416)
yololoss3 = YOLOLoss(anchors[2],4,416)
dataset = ListDateset(train_path)
dataloder  = DataLoader(dataset,batch_size=8,shuffle=True,drop_last=True)
optimizer = torch.optim.Adam(model.parameters())
model.train()
resume = False
if resume:
    chekpoint_path = './checkpoint_last_epoch.pth'
    checkpoint =  model.load(chekpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['opetimizer_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    start_epoch=0
# dataloder =tqdm(dataloder)
for epoch in range(start_epoch,100):
    totol_loss = 0
    num = 0
    for i,(_,image,target) in enumerate(dataloder):
        image = image.to(device)
        target = target.to(device)
        #Mix_up
        alpha = 1.5
        lam = np.random.beta(alpha,alpha)
        index = torch.randperm(image.size(0))
        inputs = lam*image+(1-lam)*image[index]
        x1, x2, x3 = model(inputs)
        target_a = target
        target_b = target[index]
        loss1 = lam*yololoss1(x1,target_a)+(1-lam)*yololoss1(x1,target_b)
        loss2 = lam*yololoss2(x2,target_a)+(1-lam)*yololoss2(x2,target_b)
        loss3 = lam*yololoss3(x1,target_a)+(1-lam)*yololoss3(x1,target_b)
        loss = loss1+loss2+loss3
        totol_loss+=loss
        num+=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("The loss of {} epoch {}".format(totol_loss/num,epoch))
    if epoch %10==0:
        checkpoint = {"model_state_dict":model.state_dict(),
                      "opetimizer_state_dict":optimizer.state_dict(),
                      "epoch":epoch
                      }
        save_model_path = './checkpoint_last_epoch.pth'.format(epoch)
        torch.save(checkpoint,save_model_path)

