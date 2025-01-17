import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path
import ssl


# Custom dataset class
class PedestrianDataset(Dataset):

    def __init__(self, img_dir, label_dir, transforms=None):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))
        self.labels = list(sorted(os.listdir(label_dir)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        with open(label_path, "r") as file:

            boxes = []
            labels = []

            for line in file.readlines():

                cls, x, y, w, h = map(float, line.strip().split())
                labels.append(int(cls))
                boxes.append([
                    (x - w / 2) * img.shape[1],
                    (y - h / 2) * img.shape[0],
                    (x + w / 2) * img.shape[1],
                    (y + h / 2) * img.shape[0],
                ])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# Transformations
def get_transform():
    def transform(img):
        return F.to_tensor(img)

    return transform

# Function to get the model
def get_model(num_classes):

    ssl._create_default_https_context = ssl._create_unverified_context
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Training and evaluation functions
def train_one_epoch(model, optimizer, data_loader, device):

    model.train()

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def evaluate(model, data_loader, device):

    model.eval()

    with torch.no_grad():
        total_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(data_loader)

# Main script
def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataset/")  
    train_imgs = os.path.join(data_dir, "images/train")
    train_labels = os.path.join(data_dir, "labels/train")
    val_imgs = os.path.join(data_dir, "images/val")
    val_labels = os.path.join(data_dir, "labels/val")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(device)

    train_dataset = PedestrianDataset(train_imgs, train_labels, transforms=get_transform())
    val_dataset = PedestrianDataset(val_imgs, val_labels, transforms=get_transform())

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = 1  

    # Hyperparameter
    learning_rates = [0.01, 0.0005]
    weight_decays = [0.0001, 0.0005]
    epochs = 30

    #best_model = None
    #best_loss = float("inf")

    for lr in learning_rates:
        for wd in weight_decays:

            print(f"Training with lr={lr}, wd={wd}")

            model = get_model(num_classes)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            for epoch in range(epochs):
                train_one_epoch(model, optimizer, train_loader, device)

            val_loss = evaluate(model, val_loader, device)
            

            model_save_path = f"faster_rcnn_lr_{lr}_wd_{wd}.pth"
            torch.save(model.state_dict(), model_save_path)
            

            #if val_loss < best_loss:
                #best_loss = val_loss
                #best_model = model

    

if __name__ == "__main__":
    main()