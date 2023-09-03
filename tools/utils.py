import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime
# if __name__=="__main__":
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from colorama import Fore, Style
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0 if __name__ != "__main__" else 1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from typing import Dict, Optional, Tuple, TypeVar, Union

from ultralytics import YOLO
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.ops import scale_boxes
# from utils.dataloaders import LoadImages, LoadStreams, LoadWebcam
# from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages  # , LoadStreams
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.utils.torch_utils import select_device, time_sync

from tools.StreamLoader import LoadImages, LoadStreams
from tools.transmitter import Transmitter


# FALLBACK LIST OF POSSIBLE COMBINATIONS IF FILE WITH COMBINATIONS IS NOT PROVIDED
LIST_OF_COMB = ["082","095","1204","1206","1308","1404","1405","1407","1408","1501","1503","1505","1506",
                "1508","1509","1510","1511","1516","1601","161","1602","162","163","164","1605","165","1606",
                "166","1607","1608","1609","1610","196","197","198","1910","2103","2104","2105","2106","2108",
                "1611","1612","1613","1614","1615","1616","1617","1619","1623","1625","1625","191","193","195"]
                
EPS = 1e-7
OFFSET = 30 # ,30]
def xyxy2xywh(xyxy:tuple)->tuple:
    x1,y1,x2,y2 = xyxy
    return ((x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1)
def disp_pred(pred:Union[np.ndarray,list],names:list, logger:LOGGER)->None:
    """
    Display each prediction made,
    and how many of each class is predicted.
    Args:
        pred: Predictions from the model.
        names: Names of the classes.
        logger: Logger for logging.
    """
    assert logger is not None, "Logger object is not passed"
    logger.info(f"{Fore.GREEN}Predictions:{Style.RESET_ALL}")
    class_count = np.zeros((len(names),1), dtype=np.int16)
    if len(pred):
        for j,(*xyxy, conf, cls) in enumerate(pred):
            c = int(cls)
            class_count[c] += 1
    for i, name in enumerate(names):
        if class_count[i]>0:
            logger.info(f"{Fore.GREEN}{name}:{Style.RESET_ALL} {class_count[i]}")
    logger.info(f"{Fore.GREEN}Total:{Style.RESET_ALL} {np.sum(class_count)}")
    logger.info(f"{Fore.GREEN}Most Common:{Style.RESET_ALL} {names[np.argmax(class_count)]}")
def wait_for_input(live:Union[LoadImages,LoadStreams],args:argparse.Namespace)->None:
    """
    Wait for key input from the user.
    Args:
        live: Live stream object.
    Alternatives:
        n - Next image.
        e - Exit.
        q - Finish the sequence.
        p - Previous image.
        s - speed up the pipelinen (Deactivates wait but could be stopped by pressing 's' again).
    """
    if args.wait == "skip": 
        q = cv2.waitKey(1) & 0xFF
        if q == ord('s'):
            args.wait = True
            q = cv2.waitKey(0) & 0xFF
    elif args.wait:
        wait = True

        while wait:
            q = cv2.waitKey(0) & 0xFF
            if q == ord("n"): pass
            elif q == ord("e"): exit(0) 
            elif q == ord("q"): live.count = len(live)
            elif q == ord("p"): live.count -= 2
            elif q == ord("s"):
                args.wait = "skip"
            else:
                continue
            wait = False

def visualize_yolo_2D(pred:np.ndarray,
                    img0:np.ndarray,
                    args:argparse=None,
                    names:list[str]=None,
                    rescale:bool=False,
                    img:torch.Tensor=None,
                    line_thickness:int=1, 
                    hide_labels:bool=False, 
                    hide_conf:bool=False,
                    classes_not_to_show:list[int]=None,
                    image_name:str="Object Predicitions")->str:
    """
    Visualize the predictions.\n
    Args:
        pred: Predictions from the model.\n
        pred_dict: Dictionary of predictions.\n
        img0: Image collected from the camera. # Shape: (H,W,C).\n
        img: Image predictions where based upon. # Shape: (1,3,H,W).\n
        args: Arguments from the command line.\n
        names: Names of the classes.\n
        rescale: Rescale the predictions.\n
        line_thickness: Thickness of the bounding box.\n
        hide_labels: Hide the labels.\n
        hide_conf: Hide the confidence.\n
        classes_not_to_show: Classes not to show.\n
        image_name: Name of the image.\n
    Returns:
        class_string: String of predicted classes sorted by min x value (left->right in image).\n

    """
    class_string = None
    if rescale:
        img0 = cv2.resize(img0.copy(),(640,int(640/img0.shape[1]*img0.shape[0])))
    else:
        img0 = img0.copy()
    if args is None:
        assert line_thickness is not None, "Line thickness is not passed"
        assert hide_labels is not None, "Hide labels is not passed"
        assert hide_conf is not None, "Hide confidence is not passed"
    if rescale:
        assert img is not None, "Image is not passed"
    # for i,det in enumerate(pred):
    annotator = Annotator(img0, line_width=line_thickness, example=str(names))
    if len(pred):
        if rescale:
            pred[:,:4] = scale_boxes(img.shape[2:], pred[:,:4], img0.shape[:-1]).round()
        
        classes = []
        pos_x = []
        for *xyxy, conf, cls in pred:
            c = int(cls)  # integer class
            if classes_not_to_show is not None and c in classes_not_to_show:
                continue
            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
        img0 = annotator.result()
        class_string = ""
        while len(classes)>0:
            id = np.argmin(pos_x)
            class_string += f"{names[classes[id]]}"
            pos_x.pop(id)
            classes.pop(id)
        cv2.imshow(image_name,img0)#cv2.resize(img0,(640,int(640/img0.shape[1]*img0.shape[0]))))
        cv2.waitKey(1)
    else:
        img0  = annotator.result()
        cv2.imshow(image_name,img0)#cv2.resize(img0,(640,int(640/img0.shape[1]*img0.shape[0]))))
        cv2.waitKey(1)
    return class_string
from ultralytics.yolo.engine.results import Results


def visualize_new(pred:Results, img0:np.ndarray=None,image_name:str="Object Predictions")->None:
    # plot = pred.plot(img=img0)[0].transpose(1,2,0)
    plot = pred.plot()[0].transpose(1,2,0)
    assert isinstance(plot,np.ndarray), f"plot must be a np.ndarray, got {type(plot)}"
    cv2.imshow(image_name,cv2.cvtColor(plot,cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

class TimeLogger:
    """
    Class to log time and display with pandas dataframe and matplotlib.\n
    Args:
        logger: logger object to print the time.
        disp_log: if True, display logs.
        save_log: if True, save logs. Will save to timelogger.csv under './logs/' if no name is specified.
        name: name of the timelogger.
    """
    def __init__(self,logger=None,disp_log=False,save_log=False,path="timelogger"):
        self.time_dict = {}
        self.time_pd = None
        self.metrics_pd = None
        self.logger = logger
        self.save_log = save_log
        self.path = path
        if disp_log is not None:
            self.print_log = disp_log
        else:
            self.print_log = False

    def output_log(self,name:str):
        """
        Output the time taken at each step.
        Args:
            name: name of the step, i.e. pre_process, post_process, etc.
        """
        if self.logger is not None:
            self.logger.info(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/(self.time_dict[name]['times'][-1]+EPS):.3e} Hz")
        else:
            print(f"{name}: {self.time_dict[name]['times'][-1]:.3e} s <=> {1/(self.time_dict[name]['times'][-1]+EPS):.3e} Hz")
    def create_metric(self, name: str):
        """
        Create a new metric beloning to the timelogger object.
        Args:
            name: name of the metric.
        """
        self.time_dict[name] = {}
        self.time_dict[name]["times"] = []
        self.time_dict[name]["start"] = 0
        self.time_dict[name]["stop"] = 0   
    def start(self, name: str):
        """

        """
        if name not in self.time_dict.keys():
            self.create_metric(name)
            if self.logger is not None:
                self.logger.info(f"{name} had not been initialized, initializing now.")
        self.time_dict[name]["start"] = time.monotonic()
    def stop(self, name: str):
        self.time_dict[name]["stop"] = time.monotonic()
        self.time_dict[name]["times"].append(self.time_dict[name]["stop"] - self.time_dict[name]["start"])
        if self.print_log:
            self.output_log(name)
    def log_time(self, name: str, time_: float):
        """
        Log time manually.
        Args:
            name: name of the metric.
            _time: time to log.
        """
        self.time_dict[name]["times"].append(time_)
    def maximum_time(self, name: str):
        """
        Get the maximum time taken for a step of given the name of that timing.
        Args:
            name: name of the step.
        """
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return max(self.time_dict[name]["times"])
        return 0
    def minimum_time(self, name: str):
        """
        Get the minimum time taken for a step of given the name of that timing.
        """
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return min(self.time_dict[name]["times"])
        return 0
    def average_time(self, name: str):
        """
        Get the average time taken for a step of given the name of that timing sequence.
        """
        if self.time_dict[name]["times"] is not None and len(self.time_dict[name]["times"])>0:
            return np.mean(self.time_dict[name]["times"])
        return 0
   
    def summarize(self):
        """
        Summarize the time taken for each step as well as metrics: maximum, minimum, average.
        """
        time_averages = {}
        time_max = {}
        time_min = {}
        self.time_pd = {}
        sum_ave = 0
        keys = len(self.time_dict)
        
        fig, axs = plt.subplots(keys,1)
        for i,key in enumerate(self.time_dict):
           
            if len(self.time_dict[key]["times"])>0:
                axs[i].plot(self.time_dict[key]["times"],label=key)
                axs[i].set_title(key)
                time_max[key] = self.maximum_time(key)
                time_min[key] = self.minimum_time(key)
                time_averages[key] = np.mean(np.delete(self.time_dict[key]["times"],np.argmax(self.time_dict[key]["times"])))
                sum_ave += time_averages[key] if key not in ["Full Pipeline","Internal Pipeline"] else 0
        plt.show()
        
        self.metrics_pd = pd.DataFrame([time_averages,time_max,time_min],index=["average","max","min"])
        if self.logger is not None:
            self.logger.info(f"Table To summarize:\n{self.metrics_pd}\nFrames per second: {1/self.metrics_pd['Full Pipeline']['average']:.3e} Hz")
        else:
            print(f"Table To summarize:\n{self.metrics_pd}")
        if self.save_log:
            self.metrics_pd.to_csv(f"{self.path}/timings.csv")
    
          
def get_cut_out(image:np.ndarray,xyxy:tuple,offset:Union[int,list]=OFFSET)->np.ndarray:
    """
    Get a cut out of the image.
    Args:
        image: Image from which the cut out is made.
        xyxy: Bounding box coordinates.
        offset: Offset of the cut out. Either a list of (4 or 2) or an int.
    Returns:
        cut_out: Cut out of the image.
    """
    x1,y1,x2,y2 = xyxy
    if isinstance(offset,int):
        x1 = int(x1)-offset if int(x1)-offset>=0 else 0
        y1 = int(y1)-offset if int(y1)-offset>=0 else 0
        x2 = int(x2)+offset if int(x2)+offset<image.shape[1] else image.shape[1]
        y2 = int(y2)+offset if int(y2)+offset<image.shape[0] else image.shape[0]
    else:
        if len(offset)==2:
            offset.append(offset[0])
            offset.append(offset[1])
        x1 = int(x1)-offset[0] if int(x1)-offset[0]>=0 else 0
        y1 = int(y1)-offset[1] if int(y1)-offset[1]>=0 else 0
        x2 = int(x2)+offset[2] if int(x2)+offset[2]<image.shape[1] else image.shape[1]
        y2 = int(y2)+offset[3] if int(y2)+offset[3]<image.shape[0] else image.shape[0]

    cut_out = image[int(y1):int(y2),int(x1):int(x2),:]
    return cut_out

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def create_logging_dir(run_name:str,log_dir:str,args)->str:
    """
    Create a directory for the logs.
    Args:
        run_name: Name of the run.
        log_dir: Directory where the logs are stored.
        args: Arguments of the run.
    Returns:
        log_dir: Directory where the logs are stored.
    """
    if run_name is None:
        run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(log_dir, run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Write the arguments to a file
    with open(os.path.join(log_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args, f)
    return log_dir

def scale_preds(preds:list[Union[torch.Tensor,np.ndarray]],img0:np.ndarray,img:torch.Tensor,filter:bool=False,classes_to_keep:list[int]=None)->np.ndarray:
    """
    Scale the predictions.
    Args:
        preds: Predictions of the current frame.
        img: Image predictions where made in.
        img0: Image of the current frame.
    Returns:
        preds: Scaled predictions.
    """
    if filter:
        assert classes_to_keep is not None, "classes_to_keep must be specified if filter is True"
    count = 0
    for i,pred in enumerate(preds):
        if len(pred): # if there is a detection
            pred[:,:4] = scale_coords(img.shape[2:], pred[:,:4], img0.shape).round()
            preds[i] = pred
            count += 1
    return preds#[:count]

def numpy_to_tuple(arr:np.ndarray)->tuple:
    """
    Convert a numpy array to a tuple.
    """
    return tuple(arr.tolist())

def load_img(path:str)->np.ndarray:
    """
    Load an image as a numpy array. Shape: (H,W,C) in RGB.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
def to_gray(img:np.ndarray)->np.ndarray:
    """
    Make gray scale image with RGB values.
    Args:
        img: Image to make gray scale. Shape (H,W,C).
    Returns:
        Gray scale image with 3 Channels.
    """
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def increase_contrast(img:np.ndarray)->np.ndarray:
    """
    Increase contrast of image.
    Args:
        img: Image to increase contrast of. Shape: (H,W,C) BGR.
    Returns:
        Image with increased contrast. Shape: (H,W,C) RGB.
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))

    img = clahe.apply(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img

def read_yaml(path:Union[str,Path]):
    """Read yaml file"""
    if isinstance(path, str):
        path = Path(path)
    with open(path,"r") as f:
        return yaml.safe_load(f)

def initialize_yolo_model(data, device, half, weights, imgsz):
    """
    Initialize YOLO model.
    Args:
        data: Data from the data file.
        device: Device to run the model on.
        half: Half precision.
        weights: Weights of the model.
        imgsz: Image size.
    Returns:
        YOLO model.
    """
    device = select_device(device)
    # Read names from data file.
    data_yaml = read_yaml(data["data"])
    names = data_yaml["names"]
    half = half if device.type != 'cpu' else False  # half precision only supported on CUDA
    model = YOLO(weights, task='detect')
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # tuple
    return model,imgsz,names, device

def initialize_network(args, data):
    """
    Initialize the network.
    Create live streaming object to stream data from the sensor.
    Create the detection model.
    Args:
        args: Arguments from the command line. Specified by arguments.parse_config().
    """
    model, imgsz, names, device = initialize_yolo_model(data["object"], args.device, args.half, args.weights, args.imgsz)
    # flop_counter(model, example_img) # REMOVED
    # Create file to save logs to.
    if args.save_time_log:
        dir_path = create_logging_dir(args.name_run, ROOT / "logs",args)
    else:
        dir_path = None
    source = args.source
    if source is not None and (os.path.isfile(source) or os.path.isdir(source)):
        live = LoadImages(path=source, imgsz=imgsz, vid_stride=args.vid_stride)
    elif args.webcam:
        source = args.webcam
        live = LoadStreams(sources=source, imgsz=imgsz,vid_stride=args.vid_stride)
    else:
        raise ValueError("Invalid source. Or please specify that the source is a webcam. via --webcam flag.")

                                    
    log_file = None if not args.log_all else dir_path / "full_log.txt"

    logger = create_logger(log_file=log_file)
    from ObjectTracker import RegionPredictionsTracker  # Import here to avoid circular import.
    pred_tracker = RegionPredictionsTracker(frames_to_track=args.object_frames,
                                      img_size=imgsz,
                                      threshold=args.tracker_thresh,
                                      visualize=args.visualize,
                                      class_to_track = args.class_to_track,
                                      verbose=args.verbose,
                                      logger=logger,
                                      )
    if args.transmit:
        # transmitter = Transmitter(reciever_ip=args.ip, reciever_port=args.port)
        # transmitter.start_transmit_udp()
        # transmitter.start_transmit_ml()
        raise NotImplementedError("Transmitter not implemented yet. TODO: Implement the format the data is transmitted in. (see transmitter.py, send_data() function)")
    else:
        transmitter = None
    if args.log_time:
        time_logger = TimeLogger(logger,
                                args.disp_time,
                                save_log=args.save_time_log,
                                path=dir_path)
        initialize_timer(time_logger,args)
    else:
        time_logger = None
    return model, names, device, live, pred_tracker,transmitter, time_logger, logger
def initialize_timer(time_logger:TimeLogger,args,transmitter=None):
    """
    Args:
        time_logger: The logger object to log the time taken by various parts of the pipeline.
        args: Arguments from the command line.
        transmitter: If transmitter object is available then the time taken to transmit the data is also logged.
    """
    
    time_logger.create_metric("Pre Processing")
    time_logger.create_metric("Infrence")
    time_logger.create_metric("Post Processing")
    if args.track:
        time_logger.create_metric("Tracking Frames")
    if args.visualize:
        time_logger.create_metric("Visualize")
    if args.save_csv:
        time_logger.create_metric("Save CSV")
    time_logger.create_metric("Internal Pipeline")
    time_logger.create_metric("Full Pipeline")

    return time_logger

def initialize_digit_model(args,data,logger=None):
    """
    Initialize the digit detection model with tracking.
    Args:
        args: Arguments from the command line.
    """
    model, imgsz, names, device = initialize_yolo_model(data["digit"], args.device, args.half, args.weights_digits, args.imgsz_digit)
    # flop_counter(model, example_img)
    
    from DigitDetector import DigitDetector  # Import here to avoid circular import.
    dd = DigitDetector(model=model,
                        img_size=imgsz,
                        device=device,
                        verbose=args.verbose,
                        logger=logger,
                        iou_threshold=args.iou_digits,
                        frames_to_track=args.digit_frames,
                        conf_threshold=args.conf_digits,
                        ind_threshold=args.ind_thresh,
                        seq_threshold=args.seq_thresh,
                        output_threshold=args.out_thresh,
                        list_of_combinations=args.combination_file,
                        wait=args.wait,)
    return dd, names

class ProgBar(tqdm):
    def __init__(self, is_live:bool=False,duration:Union[float,int]=-1,*args, **kwargs):
        super().__init__(*args, **kwargs,total=duration)
        self.is_live = is_live
        self.duration = duration
        self.prev_time = time.monotonic()
        self.start_time = time.monotonic()
        self._select_format()
    def _select_format(self):
        if self.is_live:
            if self.duration > 0:
                self.bar_format = '{desc}{percentage:3.0f}%|{bar:20}|'
            else:
                # Indefinite time -> No ETA
                self.bar_format = '{desc}'
        else:
            self.bar_format = '{desc}{percentage:3.0f}%|{bar:40}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    def step(self, *args, **kwargs):
        current_time = time.monotonic()
        fps = self.get_fps(current_time=current_time)
        mini_elapse = float(current_time - self.prev_time)
        elapsed = current_time - self.start_time
        if self.is_live:
            self.update(mini_elapse) # Inherited from tqdm
            # Remaining time is not available for live streams.
            if self.duration > 0:
                desc = str(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} < {time.strftime('%H:%M:%S', time.gmtime(self.duration-elapsed))}| FPS: {fps:.1f}|")
            else:
                desc =  str(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}| FPS: {fps:.1f}|")
        else:
            self.update(1)
            desc = str(f"FPS: {fps:.1f}|")
        self.prev_time = current_time
        self.set_description(desc)
        self.refresh()

    def get_fps(self,current_time):
        fps = 1/(current_time - self.prev_time+EPS)
        return fps
def test_progbar():
    _time = 10
    pb = ProgBar(is_live=False,duration=_time)
    diff1s = []
    for i in range(10):
        start1 = time.monotonic()
        pb.step()
        end1 = time.monotonic()
        diff1 = end1 - start1
        diff1s.append(diff1)
        time.sleep(0.00001)
    pb.n = pb.total
    pb.close()
    print("\n",np.mean(diff1s))
def initialize_pbar(duration,live)->ProgBar:
    pbar = ProgBar()
def norm_preds(pred,im0s):
    """
    Normalize predictions to image size.
    """
    pred[:, :4] /= np.array(im0s.shape)[[1, 0, 1, 0]]
        # if p is not None:
        #     xyxy = p[:, :4]
        #     xyxy[:, 0] /= im0s.shape[1]
        #     xyxy[:, 1] /= im0s.shape[0]
        #     xyxy[:, 2] /= im0s.shape[1]
        #     xyxy[:, 3] /= im0s.shape[0]
        #     pred[i][:,:4] = xyxy
    return pred
def flop_counter(model,x,**kwargs):
    from thop import profile
    if not isinstance(x,tuple):
        flops, params = profile(model, inputs=(x,),verbose=False)
    else:
        x,*context = x
        flops, params = profile(model, inputs=(x,*context),verbose=False)
    print(f"{model.__class__.__name__} has {flops/1e9:.2f} GFLOPS and {params/1e6:.2f} MParams")

def memory_usage(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print(f"{model.__class__.__name__} has {mem/1e6:.2f} MBytes of memory")

def int2tuple(possible_int:Union[int,list,tuple])->tuple:
    if isinstance(possible_int, int):
        possible_int = (possible_int, possible_int)
    elif isinstance(possible_int, list):
        possible_int = tuple(possible_int)
    return possible_int
def check_and_create_dir(path:str)->None:
    if not os.path.exists(path):
        os.makedirs(path)
