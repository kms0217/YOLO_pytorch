import cv2
import matplotlib.pyplot as plt
import numpy as np

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    if x_max - x_min > 0 and y_max - y_min > 0:
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name, epoch):
    img = image.copy()
    if len(bboxes) != 0 and len(category_ids) != 0:
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name)
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def getbox(model_output, S = 7, B = 2, Class_num = 20, threshold = 0.2):
    box = []
    category = []
    cell_wh = 448 / S
    # 각 Cell마다의 확률을 계산

    score = np.zeros((S * S * B, 5 + Class_num))
    for i in range(S):
        for j in range(S):
            idx = ((i * S) + j) * 2
            ox, oy, ow, oh = model_output[i][j][:4]
            xmin = (ox * cell_wh + i * cell_wh) - (ow * 448 / 2)
            ymin = (oy * cell_wh + j * cell_wh) - (oh * 448 / 2)
            xmax = xmin + (ow * 448)
            ymax = ymin + (oh * 448)
            score[idx][0] = min(xmin, xmax)
            score[idx][1] = min(ymin, ymax)
            score[idx][2] = max(xmin, xmax)
            score[idx][3] = max(ymin, ymax )
            score[idx][4] = model_output[i][j][4]

            ox, oy, ow, oh = model_output[i][j][5:9]
            xmin = (ox * cell_wh + i * cell_wh) - (ow * 448 / 2)
            ymin = (oy * cell_wh + j * cell_wh) - (oh * 448 / 2)
            xmax = xmin + (ow * 448)
            ymax = ymin + (oh * 448)
            score[idx + 1][0] = min(xmin, xmax)
            score[idx + 1][1] = min(ymin, ymax)
            score[idx + 1][2] = max(xmax, xmax)
            score[idx + 1][3] = max(ymin, ymax)
            score[idx + 1][4] = model_output[i][j][9]
            for k in range(20):
                score[idx][5 + k] = model_output[i][j][10 + k] * model_output[i][j][4]
                score[idx + 1][5 + k] = model_output[i][j][10 + k] * model_output[i][j][9]

    for i in range(Class_num):
        for j in range(98):
            if score[j][i + 5] < threshold:
                score[j][i + 5] = 0
        score = sorted(score, key = lambda x : x[i + 5], reverse=True)
        score = nms(score, i)

    for i in range(98):
        cidx = np.argmax(score[i][5:])
        if score[i][5 + cidx] == 0:
            continue
        box.append([score[i][0], score[i][1], score[i][2], score[i][3]])
        category.append(cidx)
    return box, category

def nms(score, category_idx):
    for i in range(98):
        max_score = score[i][5 + category_idx]
        if max_score == 0:
            continue
        max_score_xmin, max_score_ymin, max_score_xmax, max_score_ymax = score[i][:4]
        area1 = (max_score_xmax - max_score_xmin) * (max_score_ymax - max_score_ymin)
        for j in range(i + 1, 98):
            if score[j][5 + category_idx] == 0:
                continue
            curr_score_xmin, curr_score_ymin, curr_score_xmax, curr_score_ymax = score[j][:4]
            area2 = (curr_score_xmax - curr_score_xmin) * (curr_score_ymax - curr_score_ymin)
            inter_x = min(max_score_xmax, curr_score_xmax) - max(max_score_xmin, curr_score_xmin)
            inter_y = min(max_score_ymax, curr_score_ymax) - max(max_score_ymin, curr_score_ymin)
            inter_area = inter_x * inter_y
            union = area1 + area2 - inter_area
            IOU = 0
            if inter_x > 0 and inter_y > 0 and union > 0:
                IOU = inter_area / union
            if IOU >= 0.4:
                score[j][5 + category_idx] = 0
    return score
