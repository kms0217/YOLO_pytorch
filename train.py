from front_yolo import *
from yolo_loss import *
from yolo import *
from utils import *
from dataset import *

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.models import googlenet
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
import random

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device use : {device}")

# SEED 설정
def seed_everything(seed: int = 128):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


# Class dict
class_id = {'person' : 0, 'bird' : 1, 'cat' : 2 , 'cow' : 3 , 'dog' : 4 , 'horse' :  5, 'sheep' : 6,'aeroplane' : 7,
        'bicycle' : 8 , 'boat' : 9, 'bus' : 10, 'car' : 11, 'motorbike' : 12, 'train' : 13,
        'bottle' : 14, 'chair' : 15, 'diningtable' : 16, 'pottedplant' : 17, 'sofa' : 18, 'tvmonitor' : 19}

id_class = {0 : 'person', 1 : 'bird', 2 : 'cat' , 3 : 'cow', 4 : 'dog', 5 : 'horse', 6 :'sheep', 7 : 'aeroplane',
        8 : 'bicycle', 9 : 'boat', 10 : 'bus', 11 : 'car', 12 : 'motorbike', 13 : 'train',
        14 : 'bottle', 15 : 'chair', 16 : 'diningtable', 17 : 'pottedplant', 18 : 'sofa', 19 : 'tvmonitor'}

# train Data
transforms = A.Compose([
        A.Resize(448, 448),
        A.HueSaturationValue()
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])
)

trainset  = MyDataset("data/train",transforms)
train_iter = DataLoader(trainset, batch_size = 64, shuffle = True)

# sample image
# Data
sample_trans = A.Compose([
        A.Resize(448, 448),]
)
sample = cv2.imread("data/valid/image/2009_000412.jpg")
sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
sample = sample_trans(image = sample)
sample = sample["image"] / 255
sample_image = sample
sample = torch.Tensor(sample)
sample = sample.unsqueeze(0)

# pretrain model
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

google = googlenet(pretrained = True)
set_parameter_requires_grad(google, True)
google_conv_layer = nn.Sequential(*(list(google.children())[:-3])).to(device)

# model
yolo = Yolo(google_conv_layer).to(device)
learning_rate = 1e-2
optimizer = optim.SGD(yolo.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 0.0005)
epochs = 135

for epoch in range(epochs):
    train_loss = 0
    yolo.train()
    for idx, (img, label) in enumerate(train_iter):
        img = img.permute(0, 3, 1, 2).float()

        img, label = img.to(device), label.to(device)
        pred = yolo.forward(img)
        loss = loss_function(pred.float(), label.float(), device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    yolo.eval()
    with torch.no_grad():
        s_img = sample.permute(0, 3, 1, 2).float()
        s_img = s_img.to(device)
        s_pred = yolo.forward(s_img)
        s_pred = s_pred.detach().cpu().numpy()
        box, category = getbox(s_pred[0])
        visualize(sample_image, box, category, id_class, epoch)

    # update lr
    if (epoch == 75):
        learning_rate = 1e-3
        for p in optimizer.param_groups:
            p["lr"] = learning_rate
    if (epoch == 105):
        learning_rate = 1e-4
        for p in optimizer.param_groups:
            p["lr"] = learning_rate

    print(f"epoch : {epoch:02}, loss : {train_loss / len(train_iter):.4}")
