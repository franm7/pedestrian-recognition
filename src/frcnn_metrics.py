import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import cv2
from collections import defaultdict
from faster_rcnn import PedestrianDataset, get_transform





#Function to load the model
def load_model(model_path, num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

#Evaluate model
def evaluate_model(model, data_loader, device):

    all_preds = []
    all_targets = []
    all_scores = []

    with torch.no_grad():

        for images, targets in data_loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):

                preds = output['labels'].cpu().numpy() if 'labels' in output else []
                scores = output['scores'].cpu().numpy() if 'scores' in output else []
                true = target['labels'].cpu().numpy() if 'labels' in target else []

                if len(preds) == 0 or len(true) == 0:

                    continue

                all_preds.append(output)
                all_targets.append(target)

    return all_preds, all_targets

#Calculate metrics
def calculate_metrics(preds, targets):

    y_true = [t['labels'].cpu().numpy() for t in targets]
    y_pred = [p['labels'].cpu().numpy() for p in preds]

    precision, recall, f1, _ = precision_recall_fscore_support(
        np.concatenate(y_true), np.concatenate(y_pred), average='weighted'
    )
    return precision, recall, f1

#Calculate IoU
def calculate_iou(box_a, box_b):

    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = inter_area / (box_a_area + box_b_area - inter_area)
    return iou

#Calculate mAP@50 and mAP@50-90
def calculate_map(preds, targets):

    iou_thresholds = [x / 100 for x in range(50, 100, 5)]
    average_precisions = []

    for iou_thresh in iou_thresholds:

        tp, fp, fn = 0, 0, 0

        for pred, target in zip(preds, targets):

            pred_boxes = pred['boxes'].cpu().numpy()
            target_boxes = target['boxes'].cpu().numpy()

            matched = set()
            for p_box in pred_boxes:

                best_iou = 0
                best_t_idx = -1

                for t_idx, t_box in enumerate(target_boxes):

                    if t_idx in matched:
                        continue

                    iou = calculate_iou(p_box, t_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_t_idx = t_idx

                if best_iou >= iou_thresh:

                    tp += 1
                    matched.add(best_t_idx)

                else:

                    fp += 1

            fn += len(target_boxes) - len(matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        ap = precision * recall  
        average_precisions.append(ap)

    map_50 = average_precisions[0]
    map_50_90 = sum(average_precisions) / len(average_precisions)

    return map_50, map_50_90

# Main script
def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))  
    data_dir = os.path.join(script_dir, "dataset/")
    test_imgs = os.path.join(data_dir, "images/test")
    test_labels = os.path.join(data_dir, "labels/test")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_dataset = PedestrianDataset(test_imgs, test_labels, transforms=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


    #learning_rates = [0.01, 0.0005]
    #weight_decays = [0.0001, 0.0005]
    num_classes = 1
    model_paths = ["faster_rcnn_lr_0.01_wd_0.0001.pth", "faster_rcnn_lr_0.01_wd_0.0005.pth", 
                   "faster_rcnn_lr_0.0005_wd_0.0001.pth", "faster_rcnn_lr_0.0005_wd_0.0005.pth"]

    results_file = "frcnn_evaluation_metrics.txt"

    with open(results_file, "w") as f:

        for model_path in model_paths:

            print(f"Evaluating model: {model_path}")
            model = load_model(model_path, num_classes)
            model.to(device)

            preds, targets = evaluate_model(model, test_loader, device)

            precision, recall, f1 = calculate_metrics(preds, targets)
            map_50, map_50_90 = calculate_map(preds, targets)

            result = (f"Model: {model_path}\n"
                      f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n"
                      f"mAP@50: {map_50:.4f}, mAP@50-90: {map_50_90:.4f}\n")

            
            f.write(result + "\n")

if __name__ == "__main__":
    main()
