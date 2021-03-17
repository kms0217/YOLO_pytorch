import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def loss_function(predict, label, device, lambda_coord = 5, lambda_noobj = 0.5, S = 7, B = 2):
    '''
    args:
	predict : (batch_size, 7, 7, 30)
	label : (batch_size, 7, 7, 30)
    '''
    batch_size = predict.shape[0]
    obj_mask = label[:,:,:,4] > 0
    noobj_mask = label[:,:,:,4] == 0

    obj_mask = torch.ones_like(label) * torch.unsqueeze(obj_mask, -1)
    noobj_mask = torch.ones_like(label) * torch.unsqueeze(noobj_mask,-1)

    obj_mask = obj_mask > 0
    noobj_mask = noobj_mask > 0
    # ground에 object가 존재하는 cell
    obj_pred = predict[obj_mask].view(-1,30) # object존재하는 곳의 predict cell정보만 가져오기
    box_pred = obj_pred[:,: B * 5].contiguous().view(-1, 10) # object가 존재하는 cell에서 B개의 Box정보만
    class_pred = obj_pred[:,B * 5:].contiguous().view(-1, 20)

    obj_label = label[obj_mask].view(-1, 30)
    box_label = obj_label[:,: B * 5].contiguous().view(-1, 10)
    class_label = obj_label[:,B * 5:].contiguous().view(-1, 20)

    # ground에 object가 존재하지 않는 cell
    noobj_pred = predict[noobj_mask].view(-1,30)
    noobj_label = label[noobj_mask].view(-1,30)

    # obj가 존재하지 않는 Cell의 label의 Confidence scroe와 predict의 confidence score의 Loss
    box1_confidence_loss_noobj = F.mse_loss(noobj_pred[:,4], noobj_label[:,4], reduction = "sum")
    box2_confidence_loss_noobj = F.mse_loss(noobj_pred[:,9], noobj_label[:,9], reduction = "sum")
    confidence_loss_noobj = box1_confidence_loss_noobj + box2_confidence_loss_noobj

    # obj가 존재하는 Cell에서 IoU가 높은것만 가져오기 위한 Mask생성
    # C = pr(object) * IoU이므로 label의 C값(4, 9 index)에 IoU중 높은걸(predictor) 곱해서 계산해야한다.

    predictor_mask1 = torch.BoolTensor(box_label.shape[0])
    predictor_mask2 = torch.BoolTensor(box_label.shape[0])
    predictor_mask1.zero_()
    predictor_mask2.zero_()

    for idx, each_box in enumerate(box_pred):
        IOU1 = calc_IOU(each_box[:4], box_label[idx][:4])
        IOU2 = calc_IOU(each_box[5:9], box_label[idx][5:9])
        if IOU1 > IOU2:
            predictor_mask1[idx] = 1
        else:
            predictor_mask2[idx] = 1

    box1_pred_only_predictor = box_pred[predictor_mask1].view(-1, 10)
    box1_label_only_predictor = box_label[predictor_mask1].view(-1, 10)
    box2_pred_only_predictor = box_pred[predictor_mask2].view(-1, 10)
    box2_label_only_predictor = box_label[predictor_mask2].view(-1, 10)

    box1_confidence_loss_obj = F.mse_loss(box1_pred_only_predictor[:, 4], box1_label_only_predictor[:, 4],reduction = "sum")
    box2_confidence_loss_obj = F.mse_loss(box2_pred_only_predictor[:, 9], box2_label_only_predictor[:, 9],reduction = "sum")


    confidence_loss_obj = box1_confidence_loss_obj + box2_confidence_loss_obj

    box1_center_loss = F.mse_loss(box1_pred_only_predictor[:, :2], box1_label_only_predictor[:, :2], reduction = "sum")
    box2_center_loss = F.mse_loss(box2_pred_only_predictor[:, 5:7], box2_label_only_predictor[:, 5:7], reduction = "sum")
    center_loss = box1_center_loss + box2_center_loss

    box1_wh_loss = F.mse_loss(torch.sqrt(box1_pred_only_predictor[:, 2:4]), torch.sqrt(box1_label_only_predictor[:, 2:4]), reduction = "sum")
    box2_wh_loss = F.mse_loss(torch.sqrt(box2_pred_only_predictor[:, 7:9]), torch.sqrt(box2_label_only_predictor[:, 7:9]), reduction = "sum")
    wh_loss = box1_wh_loss + box2_wh_loss

    classification_loss = F.mse_loss(class_pred, class_label, reduction = "sum")

    return (wh_loss * lambda_coord + center_loss * lambda_coord + confidence_loss_obj + lambda_noobj * confidence_loss_noobj + classification_loss) / batch_size

def calc_IOU(predict, label):

    px, py, pw, ph = predict
    lx, ly, lw, lh = label
    predict_xmin = px / 7 - 0.5 * pw
    predict_ymin = py / 7 - 0.5 * ph
    predict_xmax = px / 7 + 0.5 * pw
    predict_ymax = py / 7 + 0.5 * ph

    label_xmin = lx / 7 - 0.5 * lw
    label_ymin = ly / 7 - 0.5 * lh
    label_xmax = lx / 7 + 0.5 * lw
    label_ymax = ly / 7 + 0.5 * lh

    predict_area = (predict_xmax - predict_xmin) * (predict_ymax - predict_ymin)
    label_area = (label_xmax - label_xmin) * (label_ymax - label_ymin)

    intersection_x_length = min(predict_xmax, label_xmax) - max(predict_xmin, label_xmin)
    intersection_y_length = min(predict_ymax, label_ymax) - max(predict_ymin, label_ymin)
    if intersection_x_length > 0 and intersection_y_length > 0:
        intersection_area = intersection_x_length * intersection_y_length
        union_area = predict_area + label_area - intersection_area
        if union_area <= 0 :
            return 0
        return intersection_area / union_area
    return 0
