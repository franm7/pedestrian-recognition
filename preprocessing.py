# Import Libraries

# Warning
import warnings
warnings.filterwarnings("ignore")

# System
import os

import glob

# Main


import pandas as pd

import cv2
from tqdm import tqdm
tqdm.pandas()
from scipy.io import loadmat

# Data Visualization
from IPython.display import Image, display, HTML

if(os.path.exists("datasets")==False):
    os.mkdir("datasets")
    
if(os.path.exists("datasets/images")==False):
    os.mkdir("datasets/images")   
if(os.path.exists("datasets/images/train")==False):
    os.mkdir("datasets/images/train")
if(os.path.exists("datasets/images/val")==False):
    os.mkdir("datasets/images/val")
    
if(os.path.exists("datasets/labels")==False):
    os.mkdir("datasets/labels")   
if(os.path.exists("datasets/labels/train")==False):
    os.mkdir("datasets/labels/train")
if(os.path.exists("datasets/labels/val")==False):
    os.mkdir("datasets/labels/val")
    
if(os.path.exists("csv_files")==False):
    os.mkdir("csv_files")



# Generate Annotations
def convertBoxFormat(box):
    (box_x_left, box_y_top, box_w, box_h) = box
    (image_w, image_h) = (640, 480)
    dw = 1./image_w
    dh = 1./image_h
    x = (box_x_left + box_w / 2.0) * dw
    y = (box_y_top + box_h / 2.0) * dh
    w = box_w * dw
    h = box_h * dh
    return (x, y, w, h)
    
annotation_dir = 'caltech/annotations/annotations/*'
classes = ['person']
number_of_truth_boxes = 0

img_id_list = []
label_list = []
split_list = []
num_annot_list = []

# Sets
for sets in tqdm(sorted(glob.glob(annotation_dir))):
    set_id = os.path.basename(sets)
    set_number = int(set_id.replace('set', ''))
    split_dataset = "train" if set_number <=5 else "val"
    
    # Videos
    for vid_annotations in sorted(glob.glob(sets + "/*.vbb")):
        video_id = os.path.splitext(os.path.basename(vid_annotations))[0] # Video ID
        vbb = loadmat(vid_annotations) # Read VBB File
        obj_lists = vbb['A'][0][0][1][0] # Annotation List
        obj_lbl = [str(v[0]) for v in vbb['A'][0][0][4][0]] # Label List
        
        # Frames
        for frame_id, obj in enumerate(obj_lists):
            if(len(obj)>0):
                # Labels
                labels = ''
                num_annot = 0
                for pedestrian_id, pedestrian_pos in zip(obj['id'][0], obj['pos'][0]):
                    pedestrian_id = int(pedestrian_id[0][0]) - 1 # Pedestrian ID
                    pedestrian_pos = pedestrian_pos[0].tolist() # Pedestrian BBox
                    # class filter and height filter: here example for medium distance
                    # if obj_lbl[pedestrian_id] in classes and pedestrian_pos[3] >= 75 and pedestrian_pos[3] <= 250:
                    if obj_lbl[pedestrian_id] in classes and pedestrian_pos[3] >= 50 and pedestrian_pos[3] <= 250:

                        yolo_box_format = convertBoxFormat(pedestrian_pos) # Convert BBox to YOLO Format
                        labels += '0 ' + ' '.join([str(n) for n in yolo_box_format]) + '\n'
                        num_annot += 1
                        number_of_truth_boxes += 1
                
                # Check Labels
                if not labels:
                    continue

                image_id = set_id + '_' + video_id + '_' + f"{frame_id:04d}"
                img_id_list.append(image_id)
                label_list.append(labels)
                split_list.append(split_dataset)
                num_annot_list.append(num_annot)
    
print("Number of Ground Truth Annotation Box:", number_of_truth_boxes)

df_caltech_annot = pd.DataFrame({
    "image_id": img_id_list,
    "label": label_list,
    "split": split_list,
    "num_annot": num_annot_list
})

df_caltech_annot["set_id"] = df_caltech_annot["image_id"].apply(lambda x: x.split("_")[0])
df_caltech_annot["video_id"] = df_caltech_annot["image_id"].apply(lambda x: x.split("_")[1])
df_caltech_annot["frame_id"] = df_caltech_annot["image_id"].apply(lambda x: int(x.split("_")[2]))

df_caltech_annot.to_csv("csv_files/frame_metadata.csv", index=False)
#df_caltech_annot


df_set_video = df_caltech_annot.groupby(["set_id", "video_id", "split"])["image_id"].count().reset_index()
df_set_video = df_set_video.rename(columns={"image_id": "total_image"})

df_set_video_train = df_set_video[df_set_video["split"]=="train"].reset_index(drop=True)
df_set_video_val = df_set_video[df_set_video["split"]=="val"].reset_index(drop=True)

display(df_set_video_train.head())
display(df_set_video_val.head())

total_train_image = sum(df_set_video_train["total_image"])
total_val_image = sum(df_set_video_val["total_image"])
print("Number of Train:", total_train_image)
print("Number of Val:", total_val_image)

df_set_video_train = df_set_video_train.groupby("set_id")["video_id"].count().reset_index()
df_set_video_val = df_set_video_val.groupby("set_id")["video_id"].count().reset_index()
df_set_video_count = pd.concat([df_set_video_train, df_set_video_val], ignore_index=True)
df_set_video_count = df_set_video_count.rename(columns={"video_id": "total_video"})
#display(df_set_video_count)


df_train_filtered = pd.DataFrame()
df_val_filtered = pd.DataFrame()

set_id_list = list(df_caltech_annot["set_id"].unique())

for i, set_id in enumerate(set_id_list):
    df_set_id = df_set_video[df_set_video["set_id"]==set_id].reset_index(drop=True)
    video_id_list = list(df_set_id["video_id"].unique())
    
    for j, vid_id in enumerate(video_id_list):
        df_video_id = df_caltech_annot[(df_caltech_annot["set_id"]==set_id) & (df_caltech_annot["video_id"]==vid_id)].reset_index(drop=True)
        frame_total = df_video_id.shape[0]
        if(i <= 5): # 20000 Train Images
            limit = int(round((frame_total / total_train_image) * 37000, 0))
            df_video_id = df_video_id[:limit]
            df_train_filtered = pd.concat([df_train_filtered, df_video_id])  # Use concat to concatenate DataFrames
        else: # 8400 Val Images
            limit = int(round((frame_total / total_val_image) * 15000, 0))
            df_video_id = df_video_id[:limit]
            df_val_filtered = pd.concat([df_val_filtered, df_video_id])  # Use concat to concatenate DataFrames

df_train_filtered = df_train_filtered.reset_index(drop=True)
df_val_filtered = df_val_filtered.reset_index(drop=True)

#display(df_train_filtered)
#display(df_val_filtered)

df_train_filtered.to_csv("csv_files/train_frame_filtered.csv", index=False)
df_val_filtered.to_csv("csv_files/val_frame_filtered.csv", index=False)


# https://github.com/simonzachau/caltech-pedestrian-dataset-to-yolo-format-converter
# Generate Images from Video Files
def save_img(dir_path, fn, i, frame):
    cv2.imwrite('{}/{}_{}_{}.png'.format(
        dir_path, os.path.basename(dir_path), os.path.basename(fn).split('.')[0], f"{i:04d}"), 
        frame)
def convert_caltech(split, df):
    print(split)
    input_dir = "caltech"
    output_dir = os.path.join("datasets", "images")
    if split == "Train":
        output_dir = os.path.join(output_dir, "train")
    else:
        output_dir = os.path.join(output_dir, "val")
    output_dir = os.path.join(output_dir, "caltechpedestriandataset")

    os.makedirs(output_dir, exist_ok=True)  # Ensure base output directory exists
    
    # Adjust glob to account for deeper structure
    sets_list = sorted(glob.glob(os.path.join(input_dir, split, "set*", "set*")))
    print("Total Sets:", len(sets_list))
    for dname in sets_list:
        print(dname)
        dname2 = os.path.basename(dname)
        output_dir2 = os.path.join(output_dir, dname2)
        os.makedirs(output_dir2, exist_ok=True)  # Use makedirs instead of mkdir

        df_filtered = df[df["set_id"] == dname2].reset_index(drop=True)
        
        # Process videos
        videos_list = list(df_filtered["video_id"].unique())
        print("Total Videos:", len(videos_list))
        for i, vd in enumerate(videos_list):
            # Adjust path to include nested setXY structure
            fn = os.path.join(dname, vd + ".seq")
            print(fn)
            cap = cv2.VideoCapture(fn)
            if not cap.isOpened():
                print(f"Failed to open file: {fn}")
                continue

            df_filtered2 = df_filtered[df_filtered["video_id"] == vd]
            frame_set = set(df_filtered2["frame_id"].unique())
            limit = len(frame_set)
            print("Total Frames:", limit)
            j = 0
            k = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if j in frame_set:
                    save_img(output_dir2, fn, j, frame)
                    k += 1
                    if k == limit:
                        break
                j += 1




convert_caltech("Train", df_train_filtered)
convert_caltech("Test", df_val_filtered)


def generate_labels(split, df):
    # Directory Path
    output_dir = "datasets/labels"
    output_dir = os.path.join(output_dir, split, "caltechpedestriandataset")
    if(os.path.exists(output_dir)==False):
        os.mkdir(output_dir)
        
    set_id_list = list(df["set_id"].unique())
    for set_id in set_id_list:
        output_dir2 = os.path.join(output_dir, set_id)
        if(os.path.exists(output_dir2)==False):
            os.mkdir(output_dir2)
        df_set_id = df[df["set_id"]==set_id].reset_index(drop=True)
        for idx, row in df_set_id.iterrows():
            label_file = open(output_dir2 + "/" + row["image_id"] + ".txt", 'w')
            label_file.write(row["label"])
            label_file.close()
            
generate_labels("train", df_train_filtered)
generate_labels("val", df_val_filtered)