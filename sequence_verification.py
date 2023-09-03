import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from colorama import Fore, Style
from random_word import RandomWords
from ultralytics.yolo.utils.ops import scale_boxes

from tools.arguments import parse_config
from tools.utils import ProgBar, disp_pred, initialize_digit_model, initialize_network, wait_for_input
from tools.visualizer import Visualizer

from typing import Tuple, Union, List, Optional, Dict

RANDOM_NAME = RandomWords()
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
EPS = 1e-8

class SequenceVerification:
    def __init__(self, args: argparse.Namespace = None):
        self.args = args
        self.init = True
        self.log_time = False
        if self.args is None:
            self.args, self.data = parse_config()
        # Setup the model, object tracker, transmitter, time logger, and logger.
        self.model, self.names_object, self.device, self.live, self.object_tracker, self.transmitter, self.time_logger, self.logger, self.dd, self.names_digit = self.setup(self.args, self.data)
        self.logger.info(f"Inference run stored @ ./logs/{self.args.name_run}")
        self.logger.info(f"Streaming data to: YoloV8 using {self.args.weights}")
        self.start_stream = time.monotonic()
        if self.args.visualize:
            self.vis_obj = Visualizer(names=self.names_object,
                                      rescale=False,
                                      line_thickness=self.args.line_thickness,
                                      hide_labels=self.args.hide_labels,
                                      hide_conf=self.args.hide_conf,
                                      image_name="Object Detector")
            if self.args.track_digits:
                self.vis_dig = Visualizer(names=self.names_digit,
                                          rescale=True,
                                          line_thickness=self.args.line_thickness,
                                          hide_labels=self.args.hide_labels,
                                          hide_conf=self.args.hide_conf,
                                          image_name="Digit Detector")
        if self.args.prog_bar:
            self.pbar = ProgBar(self.live.is_live, self.args.time)

    def setup(self, args: argparse.Namespace, data: dict) -> Tuple:
        """
        Set up the model, object tracker, transmitter, time logger, and logger.

        Args:
            args (argparse.Namespace): The command-line arguments.
            data (dict): The data configuration.

        Returns:
            tuple: The model, object names, device, live stream, object tracker, transmitter, time logger, and logger.
        """
        cudnn.benchmark = True
        model, names_object, device, live, object_tracker, transmitter, time_logger, logger = initialize_network(args, data)
        if args.track_digits:
            dd, names_digit = initialize_digit_model(args, data, logger=logger)
        else:
            dd = None
        return model, names_object, device, live, object_tracker, transmitter, time_logger, logger, dd, names_digit

    def process_image(self, img0: np.ndarray, img: torch.Tensor) -> np.ndarray:
        """
        Preprocess the image and run the inference.

        Args:
            img0 (numpy.ndarray): The original image.
            img (torch.Tensor): The preprocessed image.

        Returns:
            numpy.ndarray: The predictions.
        """
        if self.log_time:
            self.time_logger.start("Pre Processing")
        img = img.half() if self.args.half else img.float()
        if self.log_time:
            self.time_logger.stop("Pre Processing")
        if self.log_time:
            self.time_logger.start("Infrence")
        results = self.model.predict(img,
                                     augment=self.args.augment,
                                     verbose=False,
                                     nms=True,
                                     conf=self.args.conf_thres,
                                     iou=self.args.iou_thres,
                                     imgsz=img.shape[2:])[0].cpu().numpy()
        pred = results.boxes.data
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], img0.shape[:-1]).round()
        if self.log_time:
            self.time_logger.stop("Infrence")
        return pred

    def visualize(self, pred: np.ndarray, img0: np.ndarray, img: torch.Tensor) -> None:
        """
        Visualize the predictions.

        Args:
            pred (numpy.ndarray): The predictions.
            img0 (numpy.ndarray): The original image.
            img (torch.Tensor): The preprocessed image.
        """
        if self.args.disp_pred:
            disp_pred(pred, self.names_object, self.logger)
        if self.args.visualize:
            if self.log_time:
                self.time_logger.start("Visualize")
            self.vis_obj.update(pred, img0, img)
            if self.log_time:
                self.time_logger.stop("Visualize")

    def track_objects(self, pred: np.ndarray, img0: np.ndarray, img: torch.Tensor) -> None:
        """
        Track objects and digits.

        Args:
            pred (numpy.ndarray): The predictions.
            img0 (numpy.ndarray): The original image.
            img (torch.Tensor): The preprocessed image.
        """
        if self.args.track:
            if self.log_time:
                self.time_logger.start("Tracking Frames")
            best_frame = self.object_tracker.update(pred, img0, img)
            if self.log_time:
                self.time_logger.stop("Tracking Frames")
            if self.args.track_digits and (best_frame is not None or self.args.force_detect_digits):
                if self.log_time:
                    self.time_logger.start("Tracking Digit")
                img0 = best_frame["image"] if best_frame is not None else img0
                sequence, valid, result_digit, pred_digit, img = self.dd.detect(
                    img0=best_frame["image"]) if not self.args.force_detect_digits or best_frame is not None else self.dd.detect(img0)
                if self.args.visualize:
                    self.vis_dig.update(pred_digit, img0, img)
                if self.log_time:
                    self.time_logger.stop("Tracking Digit")
                if valid:
                    self.logger.info(f"Predicted Sequence: {Fore.GREEN}{sequence}{Style.RESET_ALL}\n")

    def run(self) -> None:
        """
        The main function to run the object detection.
        """
        for i, (path, img0, img, _) in enumerate(self.live):
            if not self.live.mode == "stream" and self.args.verbose:
                self.logger.info(f"Image {i}/{len(self.live)}: {path}")
            img0 = img0[0] if self.args.webcam else img0
            if self.log_time:
                self.time_logger.start("Internal Pipeline")
            if self.args.prog_bar and not self.init:
                self.pbar.step()
            pred = self.process_image(img0, img)
            self.visualize(pred, img0, img)
            self.track_objects(pred, img0, img)
            if self.args.disp_pred or self.args.verbose:
                print("\n")
            if self.init:
                self.init = False
            if self.log_time:
                self.time_logger.stop("Internal Pipeline")
            self.log_time = self.args.log_time
            if (time.monotonic() - self.start_stream) > self.args.time and self.args.time != -1:
                if self.args.prog_bar:
                    self.pbar.n = self.pbar.total
                    self.pbar.close()
                break
            if self.args.wait:
                time.sleep(0.25)
                wait_for_input(live=self.live, args=self.args)
        if self.args.transmit:
            self.transmitter.stop_transmit_udp()
            self.transmitter.stop_transmit_ml()
        if self.log_time:
            self.time_logger.summarize()
        self.logger.info("Stream Done")

if __name__ == '__main__':
    obj_detection = SequenceVerification()
    obj_detection.run()

