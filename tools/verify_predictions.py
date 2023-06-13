# from ast import Load
import os
import random
import re
import shutil
import sys
from datetime import datetime as dt
from typing import Callable

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics.yolo.utils.ops import scale_boxes
from .yolo_utils import write_label
from .arguments import parse_config
from .boliden_utils import (get_cut_out, increase_contrast, initialize_yolo_model, norm_preds, scale_preds,
                                 to_gray, visualize_yolo_2D, xyxy2xywh)
from .StreamLoader import LoadImages

TIMESTAMP = dt.now().strftime("%Y%m%d_%H%M%S")
class VerifyPredictions:
    """
    Using a Yolo model display predictions made and select wether to save image with prediction or to only save image and annotate later.
    
    Args:
        model: Yolo Model used to make predicitons.
        data: LoadImages.
        output_folder: Path where output should be stored.
    """
    def __init__(self, model, data:LoadImages, names:list, output_folder:str, count_auto_annotated=0,count_manual_annotated=0,skipped=0):
        self.model = model
        self.data = data
        self.names = names
        self.output_folder = output_folder
        self.count_auto_annotated = count_auto_annotated
        self.count_manual_annotated = count_manual_annotated
        self.skipped = skipped
        self.start = self.count_auto_annotated + self.count_manual_annotated + self.skipped
        self.data.start = self.start
        self.auto_name = "auto"
        self.manual_name = "manual"
        self.valid_list = [None]
        self.create_output_dirs()
    def create_output_dirs(self):
        """
        Create output directories.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(os.path.join(self.output_folder,self.auto_name)):
            os.makedirs(os.path.join(self.output_folder,self.auto_name))
        if not os.path.exists(os.path.join(self.output_folder,self.manual_name)):
            os.makedirs(os.path.join(self.output_folder,self.manual_name))

    def save_image_to_dir(self,img,lbls:list[torch.Tensor]=None,auto=False):
        """
        Save image to output folder.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if auto:
            cv2.imwrite(os.path.join(self.output_folder,self.auto_name, str(TIMESTAMP)+"_"+str(self.count_auto_annotated)+".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if lbls is not None:
                destination = os.path.join(self.output_folder,self.auto_name, str(TIMESTAMP)+"_"+str(self.count_auto_annotated)+".txt")
                write_label(destination, lbls, is_prediction=True)
            # with open(os.path.join(self.output_folder,self.auto_name,str(TIMESTAMP)+"_"+str(self.count_auto_annotated)+".txt"),"w") as f:
            #     for l in lbls:
            #         x1,y1,x2,y2,conf,cls = l[:6]
            #         x,y,w,h = xyxy2xywh((x1,y1,x2,y2))
            #         f.write(" ".join(str(i) for i in [int(cls),x,y,w,h])+"\n")
            self.count_auto_annotated += 1
        else:
            destination = os.path.join(self.output_folder,self.manual_name, str(TIMESTAMP)+"_"+str(self.count_manual_annotated)+".txt")
            if lbls is not None:
                write_label(destination, lbls, is_prediction=True)
                cv2.imwrite(os.path.join(self.output_folder,self.manual_name, str(TIMESTAMP)+"_"+str(self.count_manual_annotated)+".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.count_manual_annotated += 1
    def get_input(self):
        """
        Get input key press from user.
        """
        while True:
            key = cv2.waitKey(0)
            if key == ord("y"):
                return True
            elif key == ord("n"):
                return False
            elif key == ord("s"):
                return "skip"
            elif key == ord("q"):
                return "quit"
            else:
                print("Invalid key press, press 'y' to save image or 'n' to not save image labels.")
    def verify(self,pre_process:Callable=None):
        """
        Verify predictions made by model.
        """

        print("Verifying predictions...")
        pbar = tqdm(total=len(self.data))
        # Set pbar to start at current count
        pbar.update(self.start)
        winname = "Whole Image"
        cv2.namedWindow(winname) # Create a named window 
        cv2.moveWindow(winname, 40, 30)
        for path, img0, img, *_ in self.data:
            if pre_process:
                path, img0, img, *_ = pre_process(path, img0, img, *_)
            pbar.update(1)
            img0_shape = img0.shape[:-1]
            img_shape = img.shape[2:]
            results = self.model.predict(img,
                                    verbose=False,
                                    nms=True,
                                    conf=0.2,
                                    iou=0.01,
                                    max_det=6,
                                    imgsz=img_shape)[0].cpu().numpy() # Inference
            pred = results.boxes.data
            pred[:,:4] = scale_boxes(img_shape, pred[:,:4], img0_shape).round()            
            cv2.imshow(winname, cv2.resize(img0,(640,img0.shape[0]*640//img0.shape[1]))) # Resize to fit screen
            class_string = visualize_yolo_2D(pred, img0=img0, img=img,names=self.names)
            save = self.get_input()
            if save == "quit":
                print("Stopped @ Auto Annotated: {}. Manual Annotated: {}. Skipped {}".format(self.count_auto_annotated,self.count_manual_annotated,self.skipped))
                exit()
            elif save =="skip":
                self.skipped += 1
                continue
            pred = norm_preds(pred,img0)
            self.save_image_to_dir(img0,pred,save)
        print("Done verifying predictions!")
    def skip_or_false(self,false_prob:float):
        """
        Skip or return false based on probability.
        """
        det = random.random() < false_prob
        return False if det else "skip"


def verify_data(pre_process:Callable=None):        
    args, data = parse_config()
    model, imgsz, names = initialize_yolo_model(args,data)
    data = LoadImages(args.source,imgsz=imgsz)
    verify = VerifyPredictions(model,data,names,args.output_folder) # 247 141
    verify.verify(pre_process=pre_process)
if __name__=="__main__":
    verify_data(None)

# with torch.no_grad():
#     if __name__=="__main__":
#         pass
        # def pre_process(path:str, img0:np.ndarray, img:torch.Tensor, *_):
        #     # Read YOLO format labels and extract bbox of class 1
        #     label_path = path.replace(".jpg", ".txt").replace(".png", ".txt")
        #     labels = np.loadtxt(label_path, delimiter=" ", dtype=np.float32).reshape(-1, 6)
        #     labels = labels[labels[:, 0] == 1]
        #     # Extract largest bbox
        #     if len(labels):
        #         largest = np.argmax(labels[:, 4]*labels[:, 5])
        #         labels = labels[largest]
        #         cls,*xyxy = labels

        # verify.verify()
        # data_splitter = DataSplitter("../datasets/Examples/Sequence_verify/autoV2/",
        # "../datasets/YoloFormat/BolidenDigits/",0.9,0.05,0.05)
        # data_splitter.create_folders()
        # data_splitter.get_paths()
        # data_splitter.split_data()
        # # count = 0
        
        # for path, img, im0s, _,_ in tqdm(data):

        #     img = torch.from_numpy(img).to(model.device)
        #     # cv2.imshow("img",cv2.cvtColor(im0s,cv2.COLOR_RGB2BGR))
        #     # cv2.waitKey(0)
        #     img = img.float()/255.0
        #     if img.ndimension() == 3:
        #         img = img.unsqueeze(0)
        #     pred = model(img)
        #     pred = non_max_suppression(pred, 0.25, 0.45, classes=[1], agnostic=False)
        #     pred = scale_preds(pred, im0s, img)
        #     for i, det in enumerate(pred):
        #         if det is not None and len(det):
        #             # det[:, :4] = scale_preds(det[:, :4], im0s.shape)
        #             for *xyxy, conf, cls in reversed(det):
        #                 cut_out = get_cut_out(im0s, xyxy, offset=30)
        #                 cv2.imwrite("../datasets/Examples/Sequence_cut_outs/"+str(count)+".jpg", cv2.cvtColor(cut_out, cv2.COLOR_RGB2BGR))
        #                 count += 1
