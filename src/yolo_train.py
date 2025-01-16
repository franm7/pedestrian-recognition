from ultralytics import YOLO
import torch
import os

# TODO: explore and implement option for resuming model training
def train_yolo(model_version, epochs=20, batch_size=16, imgsz=640, pretrained=True):
    experiment_name = f"{model_version}_bs{batch_size}"

    if pretrained:
      model_version += ".pt"
      experiment_name += "_pretrained"
    else:
      model_version += ".yaml"
      experiment_name += "_from_scratch"

    model = YOLO(model_version)
    
    model.train(
      data="./yolo_train_config.yaml",  # Path to the dataset config file
      epochs=epochs,
      batch=batch_size,
      imgsz=imgsz,
      workers=4,
      device=0 if torch.cuda.is_available() else "cpu",
      project="runs/train",  # Output directory for training runs
      name=experiment_name
    )

    # evaluate on test dataset
    # val automatically uses parameters from train and uses model from best epoch
    # maybe make batch size bigger for test set?
    model.val(
      data="./yolo_test_config.yaml",
      project="runs/test",
      name=experiment_name
    )

if __name__ == "__main__":
    train_yolo("yolo11s", batch_size=32)

    # load model
    # base_path = os.getcwd()
    # model_path = os.path.join(base_path, "runs/train/yolo11s_bs32_pretrained/weights/best.pt")
    # model = YOLO(model_path)
    # # Perform object detection on an image
    # image_path = os.path.join(base_path, "datasets/dataset/images/test/set06_V002_0063.png")
    # results = model(image_path)
    # import matplotlib.pyplot as plt
    # import cv2

    # # Assuming results[0].plot() gives you the annotated image in OpenCV format
    # image = results[0].plot()

    # # Convert image from BGR to RGB (OpenCV loads images in BGR format)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Display using matplotlib
    # plt.imshow(image_rgb)
    # plt.axis('off')  # Hide axis
    # plt.show()