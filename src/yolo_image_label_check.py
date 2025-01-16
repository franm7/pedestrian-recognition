import os

def count_files_in_folders():
    # Use the current working directory directly
    base_path = os.getcwd()  # Ensures it starts from the root directory where you're running the script
    folders = ["train", "val", "test"]
    data_types = ["images", "labels"]

    for data_type in data_types:
        for folder in folders:
            folder_path = os.path.join(base_path, "datasets", "dataset", data_type, folder)
            if os.path.exists(folder_path):
                num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                print(f"{data_type}/{folder}: {num_files} files")
            else:
                print(f"{folder_path} does not exist.")

def check_matching_files():
    base_path = os.getcwd()  # Use current working directory
    folders = ["train", "val", "test"]

    for folder in folders:
        images_path = os.path.join(base_path, "datasets", "dataset", "images", folder)
        labels_path = os.path.join(base_path, "datasets", "dataset", "labels", folder)

        # Check if both paths exist
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Either {images_path} or {labels_path} does not exist.")
            continue

        # List files (excluding extensions for comparison)
        image_files = {f.split('.')[0] for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))}
        label_files = {f.split('.')[0] for f in os.listdir(labels_path) if f.endswith('.txt')}

        # Find mismatches
        missing_labels = image_files - label_files
        missing_images = label_files - image_files

        # Print results
        if missing_labels:
            print(f"\nImages without labels in '{folder}':")
            for file in missing_labels:
                print(f"{file}.jpg")
        if missing_images:
            print(f"\nLabels without images in '{folder}':")
            for file in missing_images:
                print(f"{file}.txt")
        if not missing_labels and not missing_images:
            print(f"\n✅ All images and labels match in '{folder}'.")

check_matching_files()
count_files_in_folders()