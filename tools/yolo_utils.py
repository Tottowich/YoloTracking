import numpy as np
import cv2
import os
import random
import colorsys
import yaml
from pathlib import Path
from typing import List, Tuple, Union
from ultralytics.yolo.utils.ops import xyxy2xywh
# Function useful fo YOLO model.

def write_label(destination:Union[str,Path], label:Union[List,np.ndarray], is_prediction:bool=False):
    """Write label to destination file"""
    if isinstance(destination, str):
        destination = Path(destination)
    if isinstance(label, List):
        label = np.array(label)
    if not destination.parent.exists():
        destination.parent.mkdir(parents=True)
    if is_prediction:
        if len(label):
            x1, y1, x2, y2, conf, cls = label.transpose()
            xyxy = np.array([x1, y1, x2, y2]).transpose()
            xywh = xyxy2xywh(xyxy)
            label = np.concatenate((cls[:,None],xywh), axis=1)
        else:
            label = np.empty((0,5))

    try:
        np.savetxt(destination, label, delimiter=' ',fmt='%d %f %f %f %f')
    except Exception as e:
        print(f"Error writing label: {label} to {destination} - {e}")

def read_label(path:Union[str,Path]):
    """Read label from path"""
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return np.empty((0,5))
    return np.loadtxt(path, delimiter=' ', dtype=np.float32).reshape(-1, 5)


def read_yaml(path:Union[str,Path]):
    """Read yaml file"""
    if isinstance(path, str):
        path = Path(path)
    with open(path,"r") as f:
        return yaml.safe_load(f)
    

def save_img(path:Union[str,Path], img:np.ndarray, RGB:bool=True):
    """Save image to path"""
    assert len(img.shape) == 3, "Image must be 3 dimensional"
    assert img.shape[-1] in [3,1], "Image must have 3 or 1 channels"
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        RGB = True
    if isinstance(path, str):
        path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if RGB:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), img)
    if not success:
        print(f"Error writing image to {path}")
    return success

def fit_label_to_img(img:np.ndarray,xywh:np.ndarray):
    shape = img.shape[:-1]
    xywh = xywh*np.array([*shape[::-1],*shape[::-1]])
    return xywh