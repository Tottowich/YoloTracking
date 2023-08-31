import cv2
import json
import yaml
import os,sys
import random
import argparse
import numpy as np
import datetime as dt
from typing import Union, Tuple
#import torchvision.transforms as transforms
"""
    Converts SuperAnnotate JSON to YOLO format
    This includes training and validation sets, 
    as well as yaml file with classes and relative directories.
    Input directory should contain:
        - images folder
        - annotations file (json)
        - classes file (json)
        - config file (json)
    Output directory will contain:
        - train folder containing images (png) and labels (txt)        
        - val folder containing images (png) and labels (txt)
        - test folder containing images (png) and labels (txt)
        - data.yaml file with classes and relative directories
"""
TIME_TAG = f"_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
def get_args():
    parser = argparse.ArgumentParser(description='Converts SuperAnnotate JSON to YOLO format')
    parser.add_argument('--input', type=str, default='input', help='input directory')
    parser.add_argument('--output', type=str, default='output', help='output folder')

    parser.add_argument('--train_val_test', nargs='+', type=float, help='Ratios per train, val, test split')
    parser.add_argument('--img_size', nargs='+', type=int, help='Image size of output image')
    parser.add_argument('--exists_ok', action='store_true', help='If set, will not create new output folder but add to existing')
    args = parser.parse_args()
    assert len(args.train_val_test) == 3, 'Train, val, test split must be provided'
    assert round(sum(args.train_val_test),3) == 1.0, 'Train, val, test split must sum to 1'
    assert len(args.img_size) in [2,1], 'Image size must be provided'
    args.img_size = [args.img_size[0], args.img_size[0]] if len(args.img_size) == 1 else args.img_size
    return args
def get_annotation_classes_paths(args):
    annotation_file = os.path.join(args.input, 'annotations.json')
    classes_file = os.path.join(args.input, 'classes.json')
    return annotation_file, classes_file

def get_classes(classes_file: str)->Tuple[list,int]:
    """
    Args:
        classes_file: 
            [
                {
                    "attribute_groups": list,
                    "color": hex,
                    "id": int, # Starts at 1
                    "name": str,
                    "opened": bool,
                },
            ]
    Output:
        classes: list of class names
        n_classes: number of classes

    """
    with open(classes_file) as f:
        data = json.load(f)
        classes = [d['name'] for d in data]
        n_classes = len(classes)
    return classes, n_classes
    
def get_annotations(annotations_file: str)->Tuple[dict,list]:
    """
    Args:
        annotations_file:
        {
            "image_name.png": {
                "instances": [
                    {
                        "type": str,
                        "classId": int,
                        "probability": int,
                        "points": {
                            "x1": float,
                            "x2": float,
                            "y1": float,
                            "y2": float
                        },
                        "groupId": int,
                        "pointLabels": {},
                        "locked": bool,
                        "visible": bool,
                        "attributes": []
                    },
                    ...
                ],
                "tags": [],
                "metadata": {
                    "version": "1.0.0",
                    "name": "image_name.png",
                    "status": "Completed"
                }
            },
        "image_name2.png": {...},
    Output: 
        annotations: dict of annotations
        image_names: list of image names
        
    """
    annotations = {}
    image_names = []
    with open(annotations_file) as f:
        data = json.load(f)
        for image_name in data:
            if not image_name.endswith('.png') and not image_name.endswith('.jpg'):
                continue
            annotations[image_name] = data[image_name]
            image_names.append(image_name)
    random.shuffle(image_names)
    return annotations, image_names

def split_image_names(image_names:list,args)->list[list]:
    """
        image_names: list of image names
        Returns: list of lists of image names
    """
    n_images = len(image_names)
    n_train = int(n_images*args.train_val_test[0]) # Number of images in train set
    n_val = int(n_images*args.train_val_test[1]) # Number of images in val set
    n_test = int(n_images*args.train_val_test[2]) # Number of images in test set
    train_names = image_names[:n_train]
    val_names = image_names[n_train:n_train+n_val]
    test_names = image_names[n_train+n_val:]
    return [train_names, val_names, test_names], [n_train, n_val, n_test]

def get_image_path(image_name:str, input_dir:str)->str:
    """
    Args:
        image_name: name of image
        input_dir: input directory
    Output:
        image_path: path to image
    """
    return os.path.join(input_dir, 'images', image_name)

def get_image(image_path:str)->np.ndarray:
    """
    Args:
        image_path: path to image
    Output:
        image: image as numpy array #(H,W,C) BGR
    """
    return cv2.imread(image_path)

def get_image_size(image:np.ndarray)->Tuple[int,int]:
    """
    Args:
        image: image as numpy array #(H,W,C) BGR
    Output:
        image_size: image size (H,W)
    """
    return image.shape[:2]

def get_image_labels(image_name:str, annotations:dict)->str:
    """
    Args:
        image_name: name of image
        annotations: dict of annotations
    Output:
        image_labels: image labels
    """
    return annotations[image_name]['instances']
def xyxy2xywh(xyxy:list)->list:
    """
        xyxy: list of [x1,y1,x2,y2]
        Returns: list of [x,y,w,h]
    """
    x1,y1,x2,y2 = xyxy
    w = x2-x1
    h = y2-y1
    x = x1+w/2
    y = y1+h/2
    return x,y,w,h
def resize_image(image:np.ndarray, img_size:list)->np.ndarray:
    """
    Args:
        image: image as numpy array #(H,W,C) BGR
        img_size: image size (H2,W2)
    Output:
        image: image as numpy array #(H2,W2,C) BGR
    """
    return cv2.resize(image, (img_size[0], img_size[1]))
def format_box_label(instances:list[dict], image_size:Tuple[int,int])->Tuple[str,list]:
    """
        instances: list of instances
        image_size: (width, height)
        Returns: string of bounding boxes in YOLO format
    """
    labels = ''
    bboxes = []
    for instance in instances:
        x1 = instance['points']['x1']/image_size[1]
        y1 = instance['points']['y1']/image_size[0]
        x2 = instance['points']['x2']/image_size[1]
        y2 = instance['points']['y2']/image_size[0]
        x,y,w,h = xyxy2xywh([x1,y1,x2,y2])
        labels += '{} {} {} {} {}\n'.format(instance['classId']-1, x, y, w, h)
        bboxes.append([x,y,w,h])
    return labels, bboxes

def get_image_labels_yolo(image_name:str, annotations:dict,input_path:str)->str:
    """
        image_name: name of current image
        annotations: dict of annotations
        Returns: string of bounding boxes in YOLO format
    """
    instances = get_image_labels(image_name, annotations)
    image_path = get_image_path(image_name, input_path)
    image = get_image(image_path)
    image_size = get_image_size(image)
    return format_box_label(instances, image_size)

def create_output_dir(output_dir:str,exists_ok:bool=False)->list[str]:
    """
    Args:
        output_dir: directory to create if it doesn't exist.
        exists_ok: if True, don't raise an error if the directory already exists.
    Output directory will contain:
        - train folder containing images (png) and labels (txt).
        - val folder containing images (png) and labels (txt).
        - test folder containing images (png) and labels (txt).
        - data.yaml file with classes and relative directories.
    """
    assert not os.path.exists(output_dir) or exists_ok and os.path.exists(output_dir), f'Output directory already exists @ {output_dir}'
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    if not exists_ok:
        os.mkdir(output_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
    return train_dir, val_dir, test_dir
def add_image(image:np.ndarray, image_name:str, output_dir:str)->None:
    """
        image: image to be saved.
        image_name: name of image.
        output_dir: directory to save image.
    """
    cv2.imwrite(os.path.join(output_dir, image_name.split('.')[0]+TIME_TAG+'.png'), image)
def add_label(label:str, image_name:str, output_dir:str)->None:
    """
        label: label to be saved.
        image_name: name of image.
        output_dir: directory to save label.
    """
    with open(os.path.join(output_dir, image_name.split('.')[0]+TIME_TAG+'.txt'), 'w') as f:
        f.write(label)

def create_data_yaml(classes_file:str,output_dir:str)->str:
    """
        classes: dict of classes.
        train_dir: directory of train images.
        val_dir: directory of val images.
        test_dir: directory of test images.
        Returns: path to data.yaml file.
    """
    classes,n_classes = get_classes(classes_file)
    data_yaml = {
        'path': output_dir,
        'train': "train",
        'val': "val",
        'test': "test",
        'nc': n_classes,
        'names': classes,
    }
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    return data_yaml_path
def show_image(image:np.ndarray, bboxes:list,img_size)->None:
    """
        image: image to be shown.
        bboxes: list of [x,y,w,h].
        Show image with bounding box.
    """
    for bbox in bboxes:
        x,y,w,h = bbox
        x = x*img_size[1]
        y = y*img_size[0]
        w = w*img_size[1]
        h = h*img_size[0]
        image = cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0,255,0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def create_dataset(args):
    """
    Create Yolo formated dataset from SuperAnnotate format.
    Args:
        args: Command line arguments.
    """
    annotation_file, classes_file = get_annotation_classes_paths(args)
    print("Annotation file: ", annotation_file)
    annotations, image_names = get_annotations(annotation_file)
    print(annotations)
    image_names_split, n_images_split = split_image_names(image_names, args)
    train_set = image_names_split[0]
    val_set = image_names_split[1]
    test_set = image_names_split[2]
    print(f"Creating dataset with {n_images_split[0]} train images, {n_images_split[1]} val images and {n_images_split[2]} test images.")
    train_dir, val_dir, test_dir = create_output_dir(args.output,args.exists_ok)
    init = False
    for set, dir in zip([train_set, val_set, test_set], [train_dir, val_dir, test_dir]):
        for image_name in set:
            image_path = get_image_path(image_name, args.input)
            image = get_image(image_path)
            labels, bboxes = get_image_labels_yolo(image_name, annotations, args.input)
            image = resize_image(image, args.img_size)
            if init:
                show_image(image, bboxes,args.img_size)
                init = False
            add_image(image, image_name, dir)
            add_label(labels, image_name, dir)
    data_yaml_path = create_data_yaml(classes_file, args.output) 
if __name__ == "__main__": # Example input: py .\superAnnotateToYolo.py --input .\datasets\SuperAnnotate\TestSVHN --output .\datasets\YoloFormat\TestSVHN --train_val_test 0.5 0.25 0.25 --img_size 448
    args = get_args()   
    create_dataset(args)

    
    
    
