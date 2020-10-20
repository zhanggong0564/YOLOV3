# from utils.yololoss import YOLOLoss
from utils.datasets import *
from util import *
from net.yolov4 import YoloBody
from torch.utils.data import DataLoader
from torchvision import transforms
from yolov3 import YOLOLoss
from utils.data_aug import *
import config
from tqdm import tqdm




def train():
    model.train()
    if resume:
        chekpoint_path = './checkpoint_last_epoch.pth'
        checkpoint =  model.load(chekpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opetimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch=0
    dataprocess =tqdm(dataloder)
    for epoch in range(start_epoch,config.epoch):
        totol_loss = 0
        num = 0
        for image,target in dataprocess:
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
            loss1 = lam * yololoss1(x1, target_a) + (1 - lam) * yololoss1(x1, target_b)
            loss2 = lam * yololoss2(x2, target_a) + (1 - lam) * yololoss2(x2, target_b)
            loss3 = lam * yololoss3(x3, target_a) + (1 - lam) * yololoss3(x3, target_b)
            loss = loss1 + loss2 + loss3
            totol_loss+=loss
            num+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dataprocess.set_description_str("epoch:{}".format(epoch))
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(loss.item()))
        print("The loss of {} epoch {}".format(totol_loss/num,epoch))
        if epoch %10==0:
            checkpoint = {"model_state_dict":model.state_dict(),
                          "opetimizer_state_dict":optimizer.state_dict(),
                          "epoch":epoch
                          }
            save_model_path = './checkpoint_last_epoch.pth'.format(epoch)
            torch.save(checkpoint,save_model_path)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloBody(len(config.anchors), config.classes).to(device)
    yololoss1 = YOLOLoss(config.anchors[0], config.classes, config.image_size)
    yololoss2 = YOLOLoss(config.anchors[1], config.classes, config.image_size)
    yololoss3 = YOLOLoss(config.anchors[2], config.classes, config.image_size)

    transform = transforms.Compose(
        [RandomHorizontalFilp(), RandomCrop(), RandomAffine(), Resize((config.image_size, config.image_size)),
         ToTensor()])
    dataset = ListDateset(config.train_path, transform=transform)
    dataloder = DataLoader(dataset, batch_size=config.batch_size,collate_fn=collate_fn, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters())
    resume = config.resume
    train()
