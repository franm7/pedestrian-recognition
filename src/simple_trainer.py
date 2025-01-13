import torch
import torch.nn as nn
from tqdm import tqdm


def get_best_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
    return device


class Trainer:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.device = (torch.device(config['training']['device'])
                       if 'device' in config['training']
                       else get_best_device())
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['model']['learning_rate']
        )

        self.bbox_criterion = nn.MSELoss()
        self.obj_criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        with tqdm(dataloader, desc="Training") as pbar:
            for images, targets in pbar:
                images = images.to(self.device)
                batch_size = images.shape[0]

                predictions = self.model(images)

                target_boxes = torch.zeros((batch_size, 4), device=self.device)
                target_obj = torch.zeros(batch_size, device=self.device)

                for idx, target in enumerate(targets):
                    if len(target['boxes']) > 0:
                        target_boxes[idx] = target['boxes'][0]
                        target_obj[idx] = 1.0

                box_loss = self.bbox_criterion(predictions[:, 1:], target_boxes)
                obj_loss = self.obj_criterion(predictions[:, 0], target_obj)

                loss = box_loss + obj_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({
                    'loss': loss.item(),
                    'box_loss': box_loss.item(),
                    'obj_loss': obj_loss.item()
                })

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Validation"):
                images = images.to(self.device)
                batch_size = images.shape[0]

                predictions = self.model(images)

                target_boxes = torch.zeros((batch_size, 4), device=self.device)
                target_obj = torch.zeros(batch_size, device=self.device)

                for idx, target in enumerate(targets):
                    if len(target['boxes']) > 0:
                        target_boxes[idx] = target['boxes'][0]
                        target_obj[idx] = 1.0

                box_loss = self.bbox_criterion(predictions[:, 1:], target_boxes)
                obj_loss = self.obj_criterion(predictions[:, 0], target_obj)
                loss = box_loss + obj_loss

                total_loss += loss.item()

        return total_loss / len(dataloader)
