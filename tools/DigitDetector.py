import os
import torch
import numpy as np
import logging
from typing import Union,Tuple
from colorama import Fore, Style
import torch
import torch.nn as nn
import numpy as np
import time
from boliden_utils import visualize_yolo_2D,numpy_to_tuple, ROOT
from ultralytics.yolo.data.augment import LetterBox
class DigitPrediction:
    """
    Class to store separate predictions of the digit detector.
    Args:
        digit: Digit predicted.
        score: Score of the prediction.
        sequence_order: Order of the prediction in the sequence.
        xyxy: Bounding box coordinates.
        img_size: Size of the image.(height,width)
    """
    def __init__(self,
                 label:int,
                 sequence_order:int,
                 score:float,
                 xyxy:tuple[int,int,int,int],
                 img_size:tuple[int,int],
                 verbose:bool=False,
                 logger:logging.Logger=None,
                 ):
        self.label = label
        self.sequence_order = sequence_order
        self.score = score
        self.xyxy = xyxy
        self.img_size = img_size
        self.verbose = verbose
        self.logger = logger
        
        if verbose:
            assert logger is not None, "logger must be specified if verbose is True"
            logger.info(f"Created DigitPrediction with label: {label}, img_size: {img_size}")
        self.rel_xyxy = self.relative_coords() 
    def relative_coords(self):
        """
        Get the relative coordinates of the bounding box.
        Args:
            None
        Returns:
            relative_coords: Relative coordinates of the bounding box.
        """
        x1,y1,x2,y2 = self.xyxy
        img_h,img_w = self.img_size
        relative_coords = (float(x1)/img_w,float(y1)/img_h,float(x2)/img_w,float(y2)/img_h)
        return relative_coords
    @property
    def x1(self)->float:
        return self.rel_xyxy[0]
    @property
    def y1(self)->float:
        return self.rel_xyxy[1]
    @property
    def x2(self)->float:
        return self.rel_xyxy[2]
    @property
    def y2(self)->float:
        return self.rel_xyxy[3]
    @property
    def w(self)->float:
        return self.x2-self.x1
    @property
    def h(self)->float:
        return self.y2-self.y1
    @property
    def area(self)->float:
        return self.w*self.h
    @property
    def center(self)->tuple[float,float]:
        return ((self.x1+self.x2)/2,(self.y1+self.y2)/2)
    @property
    def ix1(self)->int:
        return int(self.xyxy[0])
    @property
    def iy1(self)->int:
        return int(self.xyxy[1])
    @property
    def ix2(self)->int:
        return int(self.xyxy[2])
    @property
    def iy2(self)->int:
        return int(self.xyxy[3])
    @property
    def ih(self)->int:
        return self.iy2-self.iy1
    @property
    def iw(self)->int:
        return self.ix2-self.ix1
    @property
    def iarea(self)->int:
        return (self.ix2-self.ix1)*(self.iy2-self.iy1)
    @property
    def icenter(self)->tuple[int,int]:
        return ((self.ix1+self.ix2)//2,(self.iy1+self.iy2)//2)

    def __repr__(self):
        return f"DigitPrediction(label={self.label},sequence_order={self.sequence_order},score={self.score},x1={self.x1},y1={self.y1},x2={self.x2},y2={self.y2},w={self.w},h={self.h},area={self.area},center={self.center})"
    def __str__(self) -> str:
        return f"Label: {self.label} @ {self.center}, Score: {self.score:.2f}"
class DigitSequence:
    """
    Class to store the prediction sequence of the digit detector.
    Args:
        predicitons: list of torch tensors or numpy arrays containing the predictions. Shape: (N,6) where N is number of predictions and 6 corresponds to (x1,y1,x2,y2,score,class).
        img_size: Size of the image the predictions were made in.
    """
    def __init__(self,predictions:list[Union[torch.Tensor,np.ndarray]],img_size:tuple[int,int]=(448,448),verbose:bool=False,logger:logging.Logger=None):
        self.predictions = predictions
        self.img_size = img_size
        self.verbose = verbose
        self.logger = logger
        self.digit_sequence:list[DigitPrediction] = self.generate_digit_sequence()
        if verbose:
            assert logger is not None, "logger must be specified if verbose is True"
            self.logger.info(f"Created DigitSequence with {len(self.digit_sequence)} digits. Image size: {self.img_size}")
        self.sort_by_x()
    @property
    def lbl_seq(self)->list[int]:
        """
        Get the sequence of the predictions.
        Args:
            None
        Returns:
            sequence: List of the predictions.
        """
        return [digit.label for digit in self.digit_sequence]
    @property
    def int_seq(self)->int:
        """
        Get the sequence of the predictions as an integer.
        Args:
            None
        Returns:
            integer: Integer representation of the sequence.
        """
        return int(self.str_seq)
    @property
    def str_seq(self)->str:
        """
        Get the sequence of the predictions as a string.
        Args:
            None
        Returns:
            sequence_str: String of the predictions.
        """
        return "".join([str(digit.label) for digit in self.digit_sequence])
    @property
    def min_score(self)->float:
        """
        Get the minimum score of the sequence.
        Args:
            None
        Returns:
            min_sequence_score: Minimum score of the sequence.
        """
        return min([digit.score for digit in self.digit_sequence])
    @property
    def score_seq(self):
        """
        Get the sequence of the scores.
        Args:
            None
        Returns:
            sequence: List of the scores.
        """
        return [digit.score for digit in self.digit_sequence]
    @property
    def ave_score(self)->float:
        """
        Get the average score of the sequence.
        Args:
            None
        Returns:
            average_sequence_score: Average score of the sequence.
        """
        return sum([digit.score for digit in self.digit_sequence])/len(self.digit_sequence)
    def generate_digit_sequence(self)->list[DigitPrediction]:
        """
        Generate the digit sequence.
        Args:
            None
        Returns:
            digit_predictions: List of DigitPrediction objects.
        """
        digit_predictions = []
        # for i, pred in enumerate(self.predictions):
        if len(self.predictions):
            if isinstance(self.predictions,torch.Tensor):
                pred = self.predictions.cpu().numpy()
            else:
                pred = self.predictions
            for p in pred:
                digit_predictions.append(DigitPrediction(label=int(p[5]), sequence_order=None, score=float(p[4]),xyxy=numpy_to_tuple(p[:4]), img_size=self.img_size, verbose=self.verbose, logger=self.logger))
        return digit_predictions
    def sort_by_x(self):
        """
        Sort the predictions by their x coordinate in place.
        Args:
            None
        Returns:
            None
        """
        self.digit_sequence.sort(key=lambda x: x.ix1)

    def __len__(self):
        return len(self.digit_sequence)
    def __repr__(self):
        s = [f"({digit.label},{digit.score:3f})" for digit in self.digit_sequence]
        return f"DigitSequence({s}->{self.str_seq})"
class DigitPredictionTracker:
    """
    Class to track the predictions of digits.
    """
    def __init__(self,
                 frames_to_track:int=5,
                 img_size:tuple[int,int]=(448,448),
                 ind_threshold:float=0.5,
                 seq_threshold:float=0.3,
                 output_threshold:float=0.5,
                 visualize:bool=False,
                 list_of_combinations:list[DigitPrediction]=None,
                 verbose:bool=False,
                 logger:logging.Logger=None,
                 ):
        """
        Args:
            frames_to_track: Number of frames to track.
            threshold: Threshold for the combined score.
            visualize: bool, Visualize the best frame.
            list_of_combinations: List of combinations to select from.
        """
        self.verbose = verbose
        self.logger = logger
        self.img_size = img_size
        assert frames_to_track>0, "frames_to_track must be greater than 0."
        assert ind_threshold>=0, "ind_threshold must be greater than or equal to 0."
        assert seq_threshold>=0, "seq_threshold must be greater than or equal to 0."
        assert output_threshold>=0, "output_threshold must be greater than or equal to 0."
        if verbose:
            assert logger is not None, "If verbose is True, logger must be provided."
        self.frames_to_track = frames_to_track
        self.ind_threshold = ind_threshold
        self.seq_threshold = seq_threshold
        self.output_threshold = output_threshold
        self.visualize = visualize
        self.list_of_combinations = self.get_list_of_combinations(list_of_combinations)
        self.history:list[DigitSequence] = []
        self.best_sequence:DigitSequence = None
        self.best_score:float = 0
        self.best_frame:int = 0
        self.best_frame_img:np.ndarray = None
    def get_list_of_combinations(self,_list_of_combinations)->list[int]:
        """
        Get the list of combinations to select from.
        Args:
            list_of_combinations: List of combinations to select from.
        Returns:
            None
        """
        list_of_combinations = []
        if self.verbose:
            self.logger.info("Getting list of combinations...")
        if isinstance(_list_of_combinations,str):
            path = _list_of_combinations
            assert os.path.isfile(path), f"File {path} does not exist."
            try:
                with open(path, "r") as f:
                    combinations = f.readlines()
                    combinations = [x.strip().replace(" ",",").split(",") for x in combinations]
                    combinations = [y.strip("\"") for x in combinations for y in x if len(y) > 0 and y.strip("\"").isnumeric()]
            except:
                raise Exception(f"Could not read file {_list_of_combinations}.")
            list_of_combinations = combinations   
        elif isinstance(_list_of_combinations,list):
            for comb in _list_of_combinations:
                assert isinstance(comb,int) or isinstance(comb,str) and comb.isnumeric(),f"list_of_combinations must be a list of integers or strings of integers. Got {comb} of type {type(comb)}"
                if isinstance(comb,str):
                    comb = comb.strip('-./ ')
                if comb not in list_of_combinations:
                    list_of_combinations.append(str(comb))
                else:
                    if self.verbose:
                        self.logger.warning(f"Combination {comb} already in list_of_combinations. Skipping...")

        list_of_combinations.sort()
        if self.verbose:
            self.logger.info(f"List of combinations: {list_of_combinations}")
        return list_of_combinations
    def reset(self):
        """
        Reset the tracker.
        Args:
            None
        Returns:
            None
        """
        if self.verbose:
            self.logger.info("Resetting tracker...")
        self.history = []
        self.best_sequence = None
        self.best_score = 0
        self.best_frame = 0
        self.best_frame_img = None
    def sort_history(self):
        """
        Sort the history by the minimum score.
        Args:
            None
        Returns:
            None
        """
        self.history.sort(key=lambda x: x.min_score)
    def validate_sequence(self,sequence:DigitSequence)->bool:
        """
        Evaluate the sequence.
        Args:
            sequence: DigitSequence object.
        Returns:
            valid: True if the sequence is valid and should therefore be added to history.
        """
        valid = True
        if not len(sequence):
            if self.verbose:
                self.logger.warning("Sequence is empty. Skipping...")
            return False
        # Check if sequence is in list of combinations:
        
        if sequence.str_seq not in self.list_of_combinations or not len(sequence):
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence}, string: {sequence.str_seq} not in list of combinations.")
            valid = False# Must be valid sequence
        
        # Check if sequence is already in history must be the same sequence accross time:
        if sequence.int_seq not in [seq.int_seq for seq in self.history] and len(self.history)>0: # Check if not matching history sequence or if history is empty
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence} did not match history.")
            self.reset() # Must be valid sequence accross time. Resetting such that the sequence could be added if it is valid given the certainy thresholds.
        
        if sequence.min_score < self.ind_threshold:
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence} did not meet individual threshold.")
            valid = False # The minimum score must be above the threshold. Uncertainty in one digit is too high.
        
        if sequence.ave_score < self.seq_threshold:
            if self.verbose:
                self.logger.warning(f"Sequence: {sequence} did not meet sequence threshold.")
            valid = False # The average score must be above the threshold. Uncertainty in the sequence is too high.
        if valid and self.verbose:
            self.logger.info(f"Sequence: {sequence} is {Fore.GREEN}valid.{Style.RESET_ALL}")
        return valid

    def add_sequence(self,sequence:DigitSequence):
        """
        Add a sequence to the history.
        Args:
            sequence: DigitSequence object.
        Returns:
            None
        """
        if self.verbose:
            self.logger.info(f"Adding sequence: {sequence} to history...")
        self.history.append(sequence)
    def validate_history(self)->bool:
        """
        Validate the history.
        Args:
            None
        Returns:
            None
        """
        average_history_score = sum([seq.ave_score for seq in self.history])/len(self.history)
        
        if average_history_score >= self.output_threshold:
            if self.verbose:
                self.logger.info(f"Average history score: {average_history_score:.3f} is above threshold: {self.output_threshold}.-> {Fore.GREEN}Output{Style.RESET_ALL}")
            return True
        else:
            if self.verbose:
                self.logger.info(f"Average history score: {average_history_score:.3f} is below threshold: {self.output_threshold}.-> {Fore.RED}No output{Style.RESET_ALL}")
            return False
    def get_best_sequence(self)->DigitSequence:
        """
        Get the best sequence based on average score.
        Args:
            None
        Returns:
            best_sequence: DigitSequence object.
        """
        best_sequence = self.history[np.argmax([seq.ave_score for seq in self.history])]
        return best_sequence

        
        
        
    def update(self,predictions:list[Union[torch.Tensor,np.ndarray]],img:np.ndarray)->Tuple[DigitSequence,bool]:
        """
        Update the tracker.
        Args:
            predictions: List of predictions.
            img: Image the predictions were made in.
        Returns:
            None if the sequence did does not satisfy the requirements. Otherwise the sequence of numbers, i.e. the integer value of the sequence.
        """
        if self.verbose:
            self.logger.info("Updating tracker...")
        current_sequence = DigitSequence(predictions,img_size=self.img_size,verbose=self.verbose,logger=self.logger)
        if self.verbose:
            self.logger.info(f"Current sequence: {current_sequence}")
        valid = self.validate_sequence(current_sequence)
        if not valid:
            self.reset()
            return current_sequence,valid # Sequence is not valid.
        # If the average score of the history exceeds the threshold, we have an output sequence.
        # This means that the sequence must be valid over time. 
        # Criteria can be changed under validate_sequence for individual sequence added to history.
        # Criteria can be changed under validate_history for the history to be considered a valid output sequence.
        self.add_sequence(current_sequence)
        if len(self.history) > self.frames_to_track:
            self.history.pop(0)
        elif len(self.history) < self.frames_to_track:
            if self.verbose:
                self.logger.info(f"History is not long enough to be considered a valid output sequence: {len(self.history)}/{self.frames_to_track}")
            return current_sequence, False # Not enough frames to make a prediction.
        valid = best = self.validate_history()
        if best:
            best_sequence = self.get_best_sequence()
            if self.verbose:
                self.logger.info(f"Best sequence score: {best_sequence.ave_score:.3f}")
            assert isinstance(best_sequence,DigitSequence), f"best_seq must be a DigitSequence object. Got {best_sequence} of type {type(best_sequence)}"
            self.reset()
            return best_sequence,best
        else:
            if self.verbose:
                self.logger.warning(f"Sequence {current_sequence} {Fore.RED}didn't pass{Style.RESET_ALL} -> score: {current_sequence.ave_score:.3f}/{self.output_threshold:.3f}")
            return current_sequence, valid
    def __repr__(self):
        return f"DigitPredictionTracker(\n\t\tFrames={self.frames_to_track},\n\t\tIndvidual Tresh={self.ind_threshold:.3f},\n\t\tSequence Threshold={self.seq_threshold:.3f},\n\t\tOutput Threshold={self.output_threshold:.3f},\n\t\tCombinations={self.list_of_combinations})"
class DigitDetector:
    """
        Digit Detector.
        Args:
            model: A model which takes an image as input and outputs a list of predictions constiting of bounding boxes around digits, predictions should have form : x1, y1,x2,y2, conf, cls.
            device: Default to GPU.
            img_size: The size of the images the model was trained on.
            # Detection Thresholds
            iou_threshold: Intersection over union threshold.
            conf_threshold: Confidence threshold.
            # DigitPredictionTracker Parameters
            frames_to_track: Number of frames to track, i.e. how many frames to consider as history.
            ind_threshold: Threshold for individual predictions.
            seq_threshold: The average score of the sequence must be above this threshold to be considered a single valid sequence.
            output_threshold: The average score of the sequence history must be above this threshold to be a viable output.
            list_of_combinations: List of strings representing the allowed DigitSequences.
            logger: Logger object.
            visualize: Whether to visualize the predictions.
            wait: Whether to wait for a key press after showing an image.
            verbose: Whether to print information about the process.
            patience: Patience indicates how long to allow seperated predictions to be considered as same sequence.
        """
    def __init__(self,
                model:nn.Module, # A model which takes an image as input and outputs a list of predictions constiting of bounding boxes around digits, predictions should have form : x1, y1,x2,y2, conf, cls.
                device:torch.device==torch.device("cuda:0"), # Default to GPU.
                img_size:tuple[int,int]=(416,416), # The size of the images the model was trained on.
                # Detection Thresholds
                iou_threshold:float=0.5, # Intersection over union threshold.
                conf_threshold:float=0.5, # Confidence threshold.
                # DigitPredictionTracker Parameters
                frames_to_track:int=5, # Number of frames to track, i.e. how many frames to consider as history.
                ind_threshold:float=0.5, # Threshold for individual predictions.
                seq_threshold:float=0.5, # The average score of the sequence must be above this threshold to be considered a single valid sequence.
                output_threshold:float=0.5, # The average score of the sequence history must be above this threshold to be a viable output.
                list_of_combinations:list[str]=None, # List of strings representing the allowed DigitSequences.
                logger:logging.Logger=None, # Logger object.
                visualize:bool=False, # Whether to visualize the predictions.
                wait:bool=False, # Whether to wait for a key press after showing an image.
                verbose:bool=False, # Whether to print information about the process.
                patience:Union[float,int]=None, # Patience indicates how long to allow seperated predictions to be considered as same sequence.
                ):
        
        self.model = model
        self.img_size = img_size
        self.tracker = DigitPredictionTracker(frames_to_track=frames_to_track,
                                                img_size=img_size,
                                                ind_threshold=ind_threshold,
                                                seq_threshold=seq_threshold,
                                                output_threshold=output_threshold,
                                                list_of_combinations=list_of_combinations,
                                                visualize=visualize,
                                                verbose=verbose,
                                                logger=logger,
                                              )
        self.LB = LetterBox(img_size)
        self.device = device
        self.verbose = verbose
        self.logger = logger
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.classes = [0,1,2,3,4,5,6,7,8,9] # Classes should be 0-9 for digits.
        self.previous_time = 0 # Used to ensure time consistency.
        self.patience = patience
        # self.reshaper = torchvision.transforms.Resize(img_size)
        self.visualize = visualize
        self.wait = wait
        if self.verbose:
            self.logger.info("Initializing DigitDetector...")
            self.logger.info(f"Model: {self.model.__class__.__name__}")
            self.logger.info(f"Tracker: {self.tracker}")
            self.logger.info(f"Device: {self.device}")
        
    def detect(self,img0:np.ndarray)->Union[None,DigitSequence]:
        """
        Detect digits in an image.
        Args:
            img: Image to detect digits in.
        Returns:
            None if no sequence was detected. Otherwise the sequence of numbers, i.e. the integer value of the sequence.
        """

        if isinstance(img0,torch.Tensor):
            img0 = img0.numpy().transpose(1,2,0) if len(img0.shape) == 3 else img0.numpy().transpose(0,2,3,1)[0]
        img = self.pre_process(img0)
        results = self.model.predict(img, verbose=self.verbose,agnostic_nms=True, imgsz=self.img_size,conf=self.conf_threshold,iou=self.iou_threshold)[0].cpu().numpy() # Get predictions
        pred = results.boxes.data
        """
        Apply NMS, agnostic=True is used to disable class specific NMS. 
        Remove overlapping predictions no mather the class.
        """
        sequence, valid = self.tracker.update(pred, img)
        if self.verbose:
            self.logger.info(f"Sequence: {sequence} from {f'{Fore.GREEN}valid{Style.RESET_ALL}' if valid else f'{Fore.RED}invalid{Style.RESET_ALL}'} sequence.")
        return sequence, valid, results, pred, img
    def update(self,img0:np.ndarray)->Union[None,DigitSequence]:
        """
        Update the DigitDetector. Make sure that time consistency is maintained.
        Args:
            img: Image to detect digits in.
        Returns:
            None if no sequence was detected. Otherwise the sequence of numbers as a DigitSequence, see - class DigitSequence.
        """
        if self.verbose:
            self.logger.info("Updating DigitDetector...")
        current_time = time.monotonic()
        elapsed_time = current_time - self.previous_time
        self.previous_time = current_time
        if elapsed_time > self.patience:
            self.tracker.reset()
            if self.verbose:
                self.logger.info("Resetting tracker due to time inconsistency.")
        return self.detect(img0)
    def reset(self):
        """
        Reset the DigitDetector.
        """
        self.tracker.reset()

    def pre_process(self,img0:np.ndarray)->torch.Tensor:
        img = img0.copy()
        if self.verbose:
            self.logger.info("Detecting digits...")
        if isinstance(img,torch.Tensor):
            img = img.to(self.device) # Normalize
        else:
            img = self.LB(image=img)
            img = np.stack(img)
            if len(img.shape) == 3:
                img = np.expand_dims(img,axis=0)
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            img = np.ascontiguousarray(img)  # contiguous
            img = torch.from_numpy(img)
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img

    def __repr__(self):
        return f"DigitDetector(\n\t\tModel={self.model.__class__.__name__},\n\t\tDevice={self.device},\n\t\tImage size={self.img_size}\n\t\tTracker={self.tracker},\n\t\tVerbose={self.verbose})"