from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
from PIL import Image
import torch


class CaltechPedestrianDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.examples_dir = os.path.join(self.root_dir, self.split, "examples")
        self.annotations_dir = os.path.join(self.root_dir, self.split, "annotations")

        self.images = sorted(os.listdir(self.examples_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.examples_dir, image_name)
        annotation_path = os.path.join(self.annotations_dir, image_name.replace(".png", ".txt"))

        image = Image.open(image_path).convert('RGB')

        boxes = []
        labels = []

        with open(annotation_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([x_center, y_center, width, height])
                labels.append(int(class_id))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        image = self.transform(image)

        return image, {'boxes': boxes, 'labels': labels}
