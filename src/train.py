import json
from torch.utils.data import DataLoader
import torch
from dataset import CaltechPedestrianDataset
from model.SimpleDetectionModel import SimpleDetectionModel
from simple_trainer import Trainer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, 0)
    return images, targets


def main():
    config = load_config("./config.json")

    train_dataset = CaltechPedestrianDataset(
        root_dir=config['data']['path'],
        split="train"
    )

    val_dataset = CaltechPedestrianDataset(
        root_dir=config['data']['path'],
        split="val"
    )
    test_dataset = CaltechPedestrianDataset(
        root_dir=config['data']['path'],
        split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )

    model = SimpleDetectionModel()
    trainer = Trainer(model, config)

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = trainer.train_epoch(train_loader)

        val_loss = trainer.validate(val_loader)

        print(f"Train loss: {train_loss:.4f}, val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
