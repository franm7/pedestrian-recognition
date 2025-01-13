import os
import shutil

def create_folder_structure(base_path):
    """
    Create the folder structure for the new dataset.
    """
    os.makedirs(os.path.join(base_path, "train", "examples"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "train", "annotations"), exist_ok=True)

def process_train_set(input_image_dir, input_label_dir, output_dir, interval=12):
    """
    Sample every 12th image and its corresponding annotation from the source directory.
    """
    train_examples_path = os.path.join(output_dir, "train", "examples")
    train_annotations_path = os.path.join(output_dir, "train", "annotations")

    image_count = 0  # To track the current image index for sampling
    for set_folder in sorted(os.listdir(input_image_dir)):
        set_image_path = os.path.join(input_image_dir, set_folder)
        set_label_path = os.path.join(input_label_dir, set_folder)
        
        if not os.path.isdir(set_image_path):
            continue  # Skip non-directory files
        
        print(f"Processing set: {set_folder}")
        for file_name in sorted(os.listdir(set_image_path)):
            if file_name.endswith(".png"):
                if image_count % interval == 0:  # Take every 12th image
                    # Image file
                    src_image_path = os.path.join(set_image_path, file_name)
                    dest_image_path = os.path.join(train_examples_path, file_name)
                    shutil.copy2(src_image_path, dest_image_path)

                    # Corresponding annotation file
                    annotation_file = file_name.replace(".png", ".txt")
                    src_label_path = os.path.join(set_label_path, annotation_file)
                    dest_label_path = os.path.join(train_annotations_path, annotation_file)
                    if os.path.exists(src_label_path):
                        shutil.copy2(src_label_path, dest_label_path)
                    else:
                        print(f"Warning: Annotation file not found for {file_name}")

                image_count += 1  # Increment the count whether sampled or not



def process_test_set():
    # Paths
    source_images_path = "datasets/images/val/caltechpedestriandataset"
    source_labels_path = "datasets/labels/val/caltechpedestriandataset"
    destination_path = "../dataset/test"
    destination_examples_path = os.path.join(destination_path, "examples")
    destination_annotations_path = os.path.join(destination_path, "annotations")
    
    # Create directories for test set if they don't exist
    os.makedirs(destination_examples_path, exist_ok=True)
    os.makedirs(destination_annotations_path, exist_ok=True)
    
    # Only process set06 and set07
    sets_to_process = ["set06", "set07"]
    counter = 0
    
    for set_folder in sets_to_process:
        images_folder = os.path.join(source_images_path, set_folder)
        labels_folder = os.path.join(source_labels_path, set_folder)
        
        if not os.path.exists(images_folder):
            print(f"Image folder {images_folder} does not exist!")
            continue
        if not os.path.exists(labels_folder):
            print(f"Label folder {labels_folder} does not exist!")
            continue

        # Iterate through all files in the setXX folder
        image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png")])
        for i, image_file in enumerate(image_files):
            if i % 13 == 0:  # Take every 13th file
                # Define source and destination paths for images and annotations
                image_source = os.path.join(images_folder, image_file)
                annotation_source = os.path.join(labels_folder, image_file.replace(".png", ".txt"))
                
                # Destination paths
                image_destination = os.path.join(destination_examples_path, image_file)
                annotation_destination = os.path.join(destination_annotations_path, image_file.replace(".png", ".txt"))
                
                # Copy image
                shutil.copy2(image_source, image_destination)
                
                # Copy annotation only if it exists
                if os.path.exists(annotation_source):
                    shutil.copy2(annotation_source, annotation_destination)
                else:
                    print(f"Annotation file missing for {image_file}, skipping annotation.")

                counter += 1
    
    print(f"Test set processing complete. Total files copied: {counter}")


def process_val_set():
    # Paths
    source_images_path = "datasets/images/val/caltechpedestriandataset"
    source_labels_path = "datasets/labels/val/caltechpedestriandataset"
    destination_path = "../dataset/val"
    destination_examples_path = os.path.join(destination_path, "examples")
    destination_annotations_path = os.path.join(destination_path, "annotations")
    
    # Create directories for validation set if they don't exist
    os.makedirs(destination_examples_path, exist_ok=True)
    os.makedirs(destination_annotations_path, exist_ok=True)
    
    # Only process set08, set09, and set10
    sets_to_process = ["set08", "set09", "set10"]
    counter = 0
    
    for set_folder in sets_to_process:
        images_folder = os.path.join(source_images_path, set_folder)
        labels_folder = os.path.join(source_labels_path, set_folder)
        
        if not os.path.exists(images_folder):
            print(f"Image folder {images_folder} does not exist!")
            continue
        if not os.path.exists(labels_folder):
            print(f"Label folder {labels_folder} does not exist!")
            continue

        # Iterate through all files in the setXX folder
        image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(".png")])
        for i, image_file in enumerate(image_files):
            if i % 11 == 0:  # Take every 11th file
                # Define source and destination paths for images and annotations
                image_source = os.path.join(images_folder, image_file)
                annotation_source = os.path.join(labels_folder, image_file.replace(".png", ".txt"))
                
                # Destination paths
                image_destination = os.path.join(destination_examples_path, image_file)
                annotation_destination = os.path.join(destination_annotations_path, image_file.replace(".png", ".txt"))
                
                # Copy image
                shutil.copy2(image_source, image_destination)
                
                # Copy annotation only if it exists
                if os.path.exists(annotation_source):
                    shutil.copy2(annotation_source, annotation_destination)
                else:
                    print(f"Annotation file missing for {image_file}, skipping annotation.")

                counter += 1
    
    print(f"Validation set processing complete. Total files copied: {counter}")

if __name__ == "__main__":
    # Paths
    input_image_dir = "datasets/images/train/caltechpedestriandataset"
    input_label_dir = "datasets/labels/train/caltechpedestriandataset"
    output_dir = "../dataset"

    # Create folder structure
    create_folder_structure(output_dir)

    # Sample files
    #process_train_set(input_image_dir, input_label_dir, output_dir, interval=12)
    #process_test_set()
    process_val_set()

    print("Sampling completed!")
