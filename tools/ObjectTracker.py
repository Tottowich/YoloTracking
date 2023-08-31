from utils import get_cut_out
import torch
import time
import numpy as np
from colorama import Fore, Style
import logging
from typing import Union,Tuple,Dict,List
from ultralytics.yolo.utils.ops import scale_boxes
OFFSET = 30 # Tuple or int, if tuple 4 (x_top, x_bottom, y_left, y_right) offset from bounding box) 
"""
ObjectTracker class used to validate object detection accros frames and time.
"""
class RegionPredictionsTracker:
    """
    Track the 'n' previous frame's largest predictions.
    Calculates a score across the sequence of predictions made by calculating the certainty
    of the prediction and the size of the area.
    High certainty and large area = high score.
    If average score across the sequence is above a threshold, the prediction is considered a success and passed along the pipeline.\n
    Args:
        frames_to_track: number of frames to track.
        img_size: size of the image the predictions where made in.
        img0_size: size of the original image.
        threshold: threshold to determine if the prediction sequence is a success.
        visualize: if True, will display the tracked predictions.
        sequence_patience: time in seconds to allow seperated best frames may be considered part of the same sequence.

    """
    def __init__(self,
                    frames_to_track:int,
                    img_size:Union[tuple,int]=None,
                    img0_size:Union[tuple,int]=None,
                    threshold:float=0.5,
                    visualize:bool=False,
                    sequence_patience:float=5.0,
                    class_to_track:int=0,
                    verbose:bool=False,
                    logger:logging.Logger=None,
                ) -> None:
        assert frames_to_track > 0, "frames_to_track must be greater than 0."
        assert img_size is not None, "img_size must be specified."
        assert threshold <= 1, "threshold must be less or equal to 1."
        assert type(class_to_track) == int, "class_to_track must be an integer."
        self.frames_to_track = frames_to_track
        self.class_to_track = class_to_track
        self.img_size = img_size if isinstance(img_size,tuple) else (img_size,img_size) if img_size is not None else None
        self.img0_size = img0_size if isinstance(img0_size,tuple) else (img0_size,img0_size,3) if img0_size is not None else None
        self.threshold = threshold
        self.scores = []
        self.images = []
        self.prev_time = 0
        self.best_frame = None
        self.best_score = 0
        self.previous_tracked_class = None
        self.visualize = visualize
        self.sequence_patience = sequence_patience
        if verbose:
            assert logger is not None, "logger must be specified if verbose is True."
        self.verbose = verbose
        self.logger = logger
    def get_largest_area(self,pred:Union[np.ndarray,torch.Tensor],img0:np.ndarray,img:torch.Tensor)->Union[Dict[str, Union[int,float,float,np.ndarray]],None]:
        """
        From the predictions made. Return the largest area of the bounding box.
        Args:
            pred: Predictions from the model.
            img0: Original image, single frame.
        Returns:
            largest_attribute:  Dictionary containing attributes of the largest bounding box.
                                Will contain:\n
                                    - 'index': index of the largest bounding box.
                                    - 'confidence': confidence of the largest bounding box.
                                    - 'largest_area': area of the largest bounding box.
                                    - 'image': cut out of the image the largest bounding box from original image.
        """
        largest_area = None
        largest_area_index = None
        confidence_largest_area = None
        largest_attributes ={
            "index":None,
            "confidence":None,
            "largest_area":0,
            "image": None,
            "class": None,
        }
        #img0 = np.ascontiguousarray(img0)
        # pred[:,:4] = scale_boxes(self.img_size, pred[:,:4], self.img0_size).round()
        if len(pred): # if there is a detection
            for j,(*xyxy, conf, cls) in enumerate(pred):
                if int(cls) == self.class_to_track:
                    area = (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])/(self.img0_size[0]*self.img0_size[1]) # Procentage of the image size, float
                    if area > largest_attributes["largest_area"]:
                        largest_attributes["index"] = j
                        largest_attributes["confidence"] = float(conf)
                        largest_attributes["largest_area"] = float(area)
                        largest_attributes["image"] = get_cut_out(img0, xyxy, offset=OFFSET)
                        largest_attributes["class"] = int(cls)
        return largest_attributes if largest_attributes["largest_area"] > 0 else None

    def update(self,predictions:np.ndarray,img0:np.ndarray,img:torch.Tensor)->Union[None,Dict[str,Union[np.ndarray,float]]]:
        """
        Update the predictions.
        Args:
            predictions: predictions of the current frame.
            img: image predictions where made in.
            img0: image of the current frame.

        Returns:
            None: If the sequence is not long enough or the average score is below the threshold.
            best_frame: Dictionary containing the best frame and the sequence's combined score.
        """
        current_time = time.monotonic()
        self.img_size = img.shape[2:]
        self.img0_size = img0.shape[:-1]
        largest_attributes = self.get_largest_area(predictions,img0,img)
        best_frame = {}
        if largest_attributes is not None: # If there is a prediction made within the patience time
            elapased_time = current_time - self.prev_time
            self.prev_time = current_time
            if elapased_time>self.sequence_patience:
                if self.verbose:
                    self.logger.info("Sequence patience time exceeded, resetting sequence.")
                    self.reset()
            score_best = largest_attributes["confidence"] # *largest_attributes["largest_area"]
            image_selected = largest_attributes["image"]
            self.scores.append(score_best)
            self.images.append(image_selected)
            # if largest_attributes["class"]!=self.previous_tracked_class: # Reset the list if no prediction is made
            #     # print(f"Resetting the list, wrong class largest.")
            #     self.reset()
            #     self.previous_tracked_class = largest_attributes["class"]
        else:
            if self.verbose:
                self.logger.info("No prediction made.")
            self.reset()
            return None
        if len(self.scores)>self.frames_to_track: # Remove the first element of the list as if it was a queue.
            self.scores.pop(0) 
            self.images.pop(0) 
        if len(self.scores)>=self.frames_to_track: # If the sequence is long enough evaluate 
            combined_score, best_index = self.evaluate()
            if combined_score>=self.threshold:# and combined_score>=self.best_score: # If the combined score is above the threshold and is better than the previous best score
                # print(f"Combined score: {combined_score}")
                if self.verbose:
                    self.logger.info(f"Combined object score over time {Fore.GREEN}above{Style.RESET_ALL}: {combined_score:.3f}/{self.threshold:.3f}")
                best_frame["score"] = combined_score
                img0_cut = self.images[best_index]
                best_frame["image"] = img0_cut
                # self.best_score = combined_score Previous version used the combined score as the best score
                self.best_score = self.scores[best_index]
                self.best_frame = img0_cut
                self.previous_tracked_class = largest_attributes["class"]
            else:
                if self.verbose:
                    self.logger.warning(f"Combined object score is {Fore.RED}below{Style.RESET_ALL}: {combined_score:.3f}/{self.threshold:.3f}")
                best_frame = None
        else:
            if self.verbose:
                self.logger.warning(f"Object sequence is {Fore.RED}not long enough{Style.RESET_ALL}: {len(self.scores)}/{self.frames_to_track}")
            best_frame = None

        return best_frame

    def evaluate(self)->Tuple[float,int]:
        """
        Evaluate the sequence of predictions.
        Args:
            None
        Returns:
            average: Average score of the sequence.
            best_index: Index of the best frame.
        """
        average = np.mean(self.scores)
        best_index = np.argmax(self.scores)
        return average, best_index
    def __len__(self):
        return len(self.scores)
    def reset(self):
        if self.verbose:
            self.logger.info("Resetting the lists.")
        self.scores = []
        self.images = []
        self.best_frame = None
        self.best_score = 0
        self.previous_tracked_class = None
        self.prev_time = time.monotonic()