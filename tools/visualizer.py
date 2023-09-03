# import time
# from typing import List, Optional

# import cv2
# import numpy as np
# import torch
# from ultralytics.yolo.utils.ops import scale_boxes
# from ultralytics.yolo.utils.plotting import Annotator, colors


# class Visualizer:
#     def __init__(self, 
#                  names:Optional[List[str]]=None,
#                  rescale:bool=False,
#                  line_thickness:int=3, 
#                  hide_labels:bool=False, 
#                  hide_conf:bool=False,
#                  classes_not_to_show:Optional[List[int]]=None,
#                  image_name:str="Object Predicitions"):
#         self.names = names
#         self.rescale = rescale
#         self.line_thickness = line_thickness
#         self.hide_labels = hide_labels
#         self.hide_conf = hide_conf
#         self.classes_not_to_show = classes_not_to_show
#         self.image_name = image_name
#         self.last_update_time = None
#     def _initiate(self):
#         cv2.namedWindow(self.image_name, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(self.image_name, 640, 480)
#     def update(self, pred:np.ndarray, img0:np.ndarray, img:torch.Tensor=None) -> str:
#         """
#         Visualize the predictions.\n
#         Args:
#             pred: Predictions from the model.\n
#             img0: Image collected from the camera. # Shape: (H,W,C).\n
#             img: Image predictions where based upon. # Shape: (1,3,H,W).\n
#         Returns:
#             class_string: String of predicted classes sorted by min x value (left->right in image).\n

#         """

#         current_time = time.monotonic()
        
#         if self.last_update_time is not None:
#             fps = 1 / (current_time - self.last_update_time)
#             cv2.putText(img0, f'FPS: {fps:.2f}', (img0.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         self.last_update_time = current_time

#         class_string = None
#         if self.rescale:
#             img0 = cv2.resize(img0.copy(),(640,int(640/img0.shape[1]*img0.shape[0])))
#         else:
#             img0 = img0.copy()
#         annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
#         if len(pred):
#             if self.rescale:
#                 pred[:,:4] = scale_boxes(img.shape[2:], pred[:,:4], img0.shape[:-1]).round()

#             classes = []
#             pos_x = []
#             for *xyxy, conf, cls in pred:
#                 c = int(cls)  # integer class
#                 if self.classes_not_to_show is not None and c in self.classes_not_to_show:
#                     continue
#                 label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
#                 annotator.box_label(xyxy, label, color=colors(c, True))
#             img0 = annotator.result()
#             class_string = ""
#             while len(classes)>0:
#                 id = np.argmin(pos_x)
#                 class_string += f"{self.names[classes[id]]}"
#                 pos_x.pop(id)
#                 classes.pop(id)
#             cv2.imshow(self.image_name,img0)
#             cv2.waitKey(1)
#         else:
#             img0  = annotator.result()
#             cv2.imshow(self.image_name,img0)
#             cv2.waitKey(1)
#         return class_string


import time
from typing import List, Optional

import cv2
import numpy as np
import torch
from ultralytics.yolo.utils.ops import scale_boxes
from ultralytics.yolo.utils.plotting import Annotator, colors

class Visualizer:
    def __init__(self, 
                 names:Optional[List[str]]=None,
                 rescale:bool=False,
                 line_thickness:int=3, 
                 hide_labels:bool=False, 
                 hide_conf:bool=False,
                 classes_not_to_show:Optional[List[int]]=None,
                 image_name:str="Object Predicitions"):
        self.names = names
        self.rescale = rescale
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.classes_not_to_show = classes_not_to_show
        self.image_name = image_name
        self.last_update_time = None

    def _initiate(self):
        cv2.namedWindow(self.image_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.image_name, 640, 480)

    def compute_fps(self, img0):
        current_time = time.monotonic()

        if self.last_update_time is not None:
            fps = 1 / (current_time - self.last_update_time)
            cv2.putText(img0, f'FPS: {fps:.2f}', (img0.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.last_update_time = current_time

    def rescale_image(self, img0):
        if self.rescale:
            return cv2.resize(img0.copy(), (640, int(640 / img0.shape[1] * img0.shape[0])))
        else:
            return img0.copy()

    def process_predictions(self, pred, img, img0):
        class_string = None
        annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
        if len(pred):
            if self.rescale:
                pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape[:-1]).round()

            classes = []
            pos_x = []
            for *xyxy, conf, cls in pred:
                c = int(cls)  # integer class
                if self.classes_not_to_show is not None and c in self.classes_not_to_show:
                    continue
                label = None if self.hide_labels else (
                self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
            img0 = annotator.result()
            class_string = ""
            while len(classes) > 0:
                id = np.argmin(pos_x)
                class_string += f"{self.names[classes[id]]}"
                pos_x.pop(id)
                classes.pop(id)
        else:
            img0 = annotator.result()
        return img0, class_string

    def display_image(self, img0):
        cv2.imshow(self.image_name, img0)
        cv2.waitKey(1)

    def update(self, pred:np.ndarray, img0:np.ndarray, img:torch.Tensor=None) -> str:
        """
        Visualize the predictions.\n
        Args:
            pred: Predictions from the model.\n
            img0: Image collected from the camera. # Shape: (H,W,C).\n
            img: Image predictions where based upon. # Shape: (1,3,H,W).\n
        Returns:
            class_string: String of predicted classes sorted by min x value (left->right in image).\n

        """
        img0 = self.rescale_image(img0)
        self.compute_fps(img0)
        img0, class_string = self.process_predictions(pred, img, img0)
        self.display_image(img0)
        return class_string
