from ultralytics import YOLO
import torch

# TODO: explore and implement option for resuming model training
def train_yolo(model_version, epochs=20, batch_size=16, imgsz=640, pretrained=True):
    experiment_name = f"{model_version}_{batch_size}"

    if pretrained:
      model_version += ".pt"
    else:
      model_version += ".yaml"
    model = YOLO(model_version)
    
    model.train(
        data="./yolo_train_config.yaml",  # Path to the dataset config file
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=4,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs/train",  # Output directory for training runs, maybe change this?
        name=experiment_name # Experiment name
    )

    # redundant? just does validation from the best epoch
    # metrics = model.val() # val automatically uses parameters from train and uses model from best epoch
    # print(metrics)

    # evaluate on test dataset
    # maybe make batch size bigger for test set?
    model.val(data="./yolo_test_config.yaml")

    # model.save('yolo_pedestrian.pt') # don't need this?

if __name__ == "__main__":
    train_yolo("yolov8s")
