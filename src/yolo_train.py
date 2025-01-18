from ultralytics import YOLO
import torch
import os
import csv

# TODO: explore and implement option for resuming model training
def train_yolo(model_name, epochs=30, batch_size=16, imgsz=640, pretrained=True, evaluate_on_test=True):
    experiment_name = f"{model_name}_bs{batch_size}"
    experiment_name += "_pretrained" if pretrained else "_from_scratch"
    model_version = model_name + (".pt" if pretrained else ".yaml")

    model = YOLO(model_version)
    
    model.train(
      data="./yolo_config.yaml",  # Path to the dataset config file
      epochs=epochs,
      batch=batch_size,
      imgsz=imgsz,
      workers=4,
      device=0 if torch.cuda.is_available() else "cpu",
      pretrained=pretrained,
      project="runs/train",  # Output directory for training runs
      name=experiment_name
    )

    if evaluate_on_test:
      evaluate_yolo(model, model_name, split="test", batch_size=batch_size, imgsz=imgsz, pretrained=pretrained)

def evaluate_yolo(model, model_name, split="test", batch_size=16, imgsz=640, pretrained=True, save_json=True, save_plots=True):
    experiment_name = f"{model_name}_bs{batch_size}"
    experiment_name += "_pretrained" if pretrained else "_from_scratch"

    metrics = model.val(
      data="./yolo_config.yaml",
      batch=batch_size,
      imgsz=imgsz,
      split=split,
      save_json=save_json,
      plots=save_plots,
      project=f"runs/{split}",
      name=experiment_name
    )

    results_data = [
      ["Metric", "Value"],
      ["Precision", metrics.box.p],
      ["Recall", metrics.box.r],
      ["mAP@50", metrics.box.map50],
      ["mAP@50-95", metrics.box.map]
    ]

    results_file = f"./runs/{split}/{experiment_name}/results.csv"
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results_data)
    
    print(f"Results of {split} set successfully saved to: {results_file}")

if __name__ == "__main__":
    # fine tune the model
    search_space = {
      "lr0": (1e-6, 1e-1),
      "degrees": (0.0, 45.0),
    }

    model = YOLO("yolo11s.pt")
    lr_results = model.tune(
        data="./yolo_config.yaml",
        epochs=30,
        iterations=10, # or more probably
        space=search_space,
        optimizer="AdamW", # or SGD myb
        fraction=0.1,  # Use 10% of the dataset (set to more myb)
        plots=True,
        save=True,
        val=False
    )

    print(f"Recommended learning rate: {lr_results['lr']}")
    print(f"Results: {lr_results}")

    # train_yolo("yolo11s", batch_size=32)
    
    # load model
    # base_path = os.getcwd()
    # model_path = os.path.join(base_path, "./runs/train/yolo11s_bs32_pretrained/weights/best.pt")
    # model_path = r"C:\Users\Adam\Downloads\runs\runs\train\yolo11s_bs48_pretrained2\weights\best.pt"
    # model = YOLO(model_path)

    # evaluate_yolo(model, "yolo11s", batch_size=32)

    # # Perform object detection on a specific image
    # base_path = os.getcwd()
    # image_path = os.path.join(base_path, r"src\set07_V010_0516.png")
    # results = model.predict(image_path, )
    # results[0].show()