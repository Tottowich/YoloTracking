# Script to extract the labels from the YOLO format and save them as images.
#!/usr/bin/env python3
import os
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.yolo_utils import read_label, save_img, fit_label_to_img
from tools.utils import check_and_create_dir, get_cut_out
from ultralytics.yolo.utils.ops import xywh2xyxy
class LabelExtractor:
    def __init__(self, input_folder: str, output_folder: str, classes_to_extract:list[int]=None) -> None:
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.data_paths = []
        self.classes_to_extract = classes_to_extract
        self.counter = 0
        self.get_paths()
        check_and_create_dir(self.output_folder)
    def get_paths(self):
        """
        Get paths to images and labels. Store them in tuple.
        """
        for file in os.listdir(self.input_folder):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(self.input_folder,file)
                label_path = os.path.join(self.input_folder,file.split(".")[0]+".txt")
                self.data_paths.append((img_path,label_path))

    def extract_labels(self):
        for file in self.data_paths:
            img_path, label_path = file
            img = cv2.imread(img_path)
            labels = read_label(label_path)
            for label in labels:
                cls, *xywh = label
                if self.classes_to_extract is not None and cls not in self.classes_to_extract:
                    continue
                xywh = fit_label_to_img(img, np.array(xywh))
                xyxy = xywh2xyxy(xywh)
                cut_out = get_cut_out(img, xyxy)
                save_img(os.path.join(self.output_folder, str(self.counter)+".jpg"), cut_out)
                self.counter += 1
            break


if __name__=="__main__":
    input_folder = "/home/thjo/Datasets/Schenk/test/"
    output_folder = "./test_folder/"
    classes_to_extract = [1]
    LE = LabelExtractor(input_folder,output_folder,classes_to_extract)
    LE.extract_labels()
    