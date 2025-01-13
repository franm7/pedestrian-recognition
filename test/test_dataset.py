import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import torch

import sys
from os.path import dirname, join, abspath
import random

# import iz src
sys.path.append(abspath(join(dirname(__file__), '..')))
from src.dataset import CaltechPedestrianDataset
from src.helper import yolo_to_pixel


def get_random_color():
    return random.random(), random.random(), random.random()


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, 0)
    return images, targets


def test_dataset():
    dataset = CaltechPedestrianDataset(root_dir="dataset", split="train")

    image, target = dataset[0]
    print("\nSingle item test:")
    print(f"Image shape: {image.shape}")
    print(f"Number of boxes: {len(target['boxes'])}")
    print(f"Number of labels: {len(target['labels'])}")

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    batch = next(iter(dataloader))
    print("\nBatch test:")
    print(f"Batch image shape: {batch[0].shape}")
    print(f"Batch targets length: {len(batch[1])}")

    def visualize_sample(image, boxes, labels, idx):
        img = image.permute(1, 2, 0).numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        height, width = img.shape[:2]

        class_names = {
            0: 'pedestrian',
        }

        for box, label in zip(boxes, labels):
            x_center, y_center, w, h = box

            color = get_random_color()

            x1, y1, x2, y2 = yolo_to_pixel(x_center, y_center, w, h, width, height)

            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], '-', color=color, linewidth=2)

            label_text = class_names.get(label.item(), f'class_{label.item()}')
            plt.text(x1, y1 - 5, label_text, color=color,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))

        plt.title(f'Image #{idx}')
        plt.axis('off')
        plt.show()

    images, targets = next(iter(dataloader))
    for idx, (image, target) in enumerate(zip(images, targets)):
        visualize_sample(image, target['boxes'], target['labels'], idx)


if __name__ == "__main__":
    test_dataset()
