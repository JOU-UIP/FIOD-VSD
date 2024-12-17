import json
import os
import cv2
import glob
from tqdm import tqdm
from mytools import tool_0

def get_clsLabel(name):
    cls0_List, cls1_List, cls2_List = tool_0()
    cls0_aug = glob.glob('/home/underwater/Code/Data/DeepfishDomain/0_aug/images/*.jpg')
    cls0_aug_List = [os.path.basename(path) for path in cls0_aug]
    cls0_List = cls0_List + cls0_aug_List
    if name in cls0_List:
        return 0
    elif name in cls1_List:
        return 1
    elif name in cls2_List:
        return 2
    else:
        raise


def per_Deepfish2COCO(baseName):
    # Define the path to the train.txt file
    train_txt_path = f'/home/underwater/Code/Data/DeepfishDomain/annotations/{baseName}.txt'

    # Define the path to the directory where the images and labels are stored
    data_dir = '/home/underwater/Code/Data/DeepfishDomain'

    # Initialize the dictionary to store the coco format annotations
    coco_dict = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define the categories
    categories = [
        {"id": 1, "name": "fish", "supercategory": "object"}
    ]

    # Add the categories to the coco dictionary
    coco_dict["categories"] = categories

    # Read the train.txt file
    with open(train_txt_path, "r") as f:
        lines = f.readlines()
        filenames = [line.strip() for line in lines]

    # Loop through the filenames and create the coco annotations
    image_id = 1
    annotation_id = 1
    for filename in filenames:
        if 'no_fish'in filename or 'NF' in filename:
            continue
        # Get the image width and height
        image_path = os.path.join(data_dir, 'images', filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        cls_label = get_clsLabel(filename)

        # Add the image to the coco dictionary
        image_dict = {
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "cls": cls_label
        }
        coco_dict["images"].append(image_dict)

        # Read the label file
        label_path = os.path.join(data_dir, 'labels', filename.split('.')[0] + '.txt')
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Loop through the label lines and create the coco annotations
        for line in lines:
            bbox = line.strip().split()[1:]
            if bbox==[]:
                continue
            x, y, w, h = map(float, bbox)
            x=x*width
            w=w*width
            y=y*height
            h=h*height
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            # Add the annotation to the coco dictionary
            annotation_dict = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            coco_dict["annotations"].append(annotation_dict)

            # Increment the annotation id
            annotation_id += 1

        # Increment the image id
        image_id += 1

    # Write the coco dictionary to a json file
    with open(os.path.join('/home/underwater/Code/Data/DeepfishDomain/annotations', f'box_{baseName}.json'), 'w') as f:
        json.dump(coco_dict, f)

if __name__=="__main__":
    txtlist = glob.glob('/home/underwater/Code/Data/DeepfishDomain/annotations/*.txt')
    for txt in tqdm(txtlist):
        baseName=os.path.basename(txt).split('.')[0]
        per_Deepfish2COCO(baseName)