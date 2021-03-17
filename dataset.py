import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from torch.autograd import Variable
import xml.etree.ElementTree as elemTree
import albumentations as A

# Class dict
class_id = {'person' : 0, 'bird' : 1, 'cat' : 2 , 'cow' : 3 , 'dog' : 4 , 'horse' :  5, 'sheep' : 6,'aeroplane' : 7,
        'bicycle' : 8 , 'boat' : 9, 'bus' : 10, 'car' : 11, 'motorbike' : 12, 'train' : 13,
        'bottle' : 14, 'chair' : 15, 'diningtable' : 16, 'pottedplant' : 17, 'sofa' : 18, 'tvmonitor' : 19}

id_class = {0 : 'person', 1 : 'bird', 2 : 'cat' , 3 : 'cow', 4 : 'dog', 5 : 'horse', 6 :'sheep', 7 : 'aeroplane',
        8 : 'bicycle', 9 : 'boat', 10 : 'bus', 11 : 'car', 12 : 'motorbike', 13 : 'train',
        14 : 'bottle', 15 : 'chair', 16 : 'diningtable', 17 : 'pottedplant', 18 : 'sofa', 19 : 'tvmonitor'}

class MyDataset(Dataset):
    '''
    Dataset 정의
    label :
        Pascal Voc Data set의 annotation에서 class, xmin, ymin, xmax, ymax를 yolo에서
        정의한 prediction과 같은 구조로 바꿔서 반환
    '''
    def __init__(self, dir, transforms = None, S = 7, B = 2, class_num = 20):
        self.dir = dir
        self.S = S
        self.B = B
        self.class_num = 20
        self.img_list = os.listdir(os.path.join(dir, "Image"))
        self.ano_list = os.listdir(os.path.join(dir, "Annotation"))
        self.img_list.sort()
        self.ano_list.sort()
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_id = self.img_list[index]
        image = cv2.imread(os.path.join(self.dir, "Image",image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = elemTree.parse(os.path.join(self.dir, "Annotation",self.ano_list[index]))
        root = annotation.getroot()
        category = []
        bboxes = []
        for obj in root.iter("object"):
            box = []
            category.append(class_id[obj.findtext("name")])
            box.append(float(obj.find("bndbox").findtext("xmin")))
            box.append(float(obj.find("bndbox").findtext("ymin")))
            box.append(float(obj.find("bndbox").findtext("xmax")))
            box.append(float(obj.find("bndbox").findtext("ymax")))
            bboxes.append(box)
        after_trans = self.transforms(image=image, bboxes = bboxes, category_ids = category)

        grid_wh = int(448 / self.S)
        label = np.zeros((self.S, self.S, 5 * self.B + self.class_num))
        for idx in range(len(after_trans["category_ids"])):
            xmin, ymin, xmax, ymax = after_trans["bboxes"][idx]
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            xmid, ymid = (xmax + xmin) / 2, (ymax + ymin) / 2
            i, j = int(xmid / grid_wh), int(ymid / grid_wh)
            x, y = (xmid - i * grid_wh) / grid_wh, (ymid - j * grid_wh) / grid_wh
            w, h = (xmax - xmin) / 448, (ymax - ymin) / 448
            temp = [x, y, w, h, 1, x, y, w, h, 1]
            temp1 = [1.0 if index == after_trans["category_ids"][idx] else 0.0 for index in range(20)]
            label[i][j] = temp + temp1
        img = after_trans["image"] / 255
        return img, label
