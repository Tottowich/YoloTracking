import argparse
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import time
from pathlib import Path
from copy import deepcopy
import torch
import numpy as np
from colorama import Fore, Style
from random_word import RandomWords

from tools.arguments import parse_config
from ultralytics.yolo.utils.ops import scale_boxes
RANDOM_NAME = RandomWords()
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
import torch.backends.cudnn as cudnn

from tools.utils import (ProgBar, disp_pred, initialize_digit_model, initialize_network, initialize_timer,
                                 scale_preds, wait_for_input, visualize_yolo_2D, visualize_new)
# from tools.visualization import visualize_yolo_2D
#PYTORCH_ENABLE_MPS_FALLBACK=1
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
EPS = 1e-8

@torch.no_grad() # No grad to save memory
def main(args: argparse.Namespace=None) -> None:
    if args is None:
        args, data = parse_config()
    init = True
    cudnn.benchmark = True  # set True to speed up constant image size inference
    model, names_object, device, live, object_tracker, transmitter, time_logger, logger = initialize_network(args,data)
    # Transmitter is currently unused
    if args.track_digits:
        dd, names_digit = initialize_digit_model(args,data,logger=logger)
    else:
        dd = None
    
    log_time = False # False to let the program run for one loop to warm up :)
    logger.info(f"Infrence run stored @ ./logs/{args.name_run}")
    logger.info(f"Streaming data to: Yolov5 using {args.weights}")
    start_stream = time.monotonic()

    if args.prog_bar:
        pbar = ProgBar(live.is_live, args.time)
    for i,(path, img0, img,_) in enumerate(live):
        if not live.mode=="stream" and args.verbose:
            logger.info(f"Image {i}/{len(live)}: {path}")
        img0 = img0[0] if args.webcam else img0
        img0_shape = img0.shape[:-1]
        img_shape = img.shape[2:]
        assert isinstance(img0, np.ndarray), f"img0 must be a np.ndarray, got {type(img0)}"
        assert isinstance(img, torch.Tensor), f"img must be a torch.Tensor, got {type(img)}"
        if log_time:
            time_logger.start("Internal Pipeline")
        if args.prog_bar and not init:
            if args.webcam:
                pbar.step()
            else:
                pbar.step()
        if log_time:
            time_logger.start("Pre Processing")
        img = img.half() if args.half else img.float()  # uint8 to fp16/32

        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if init:
            t1 = time.monotonic()
            if args.verbose:
                # logger.info(f"Image size: {img.shape}")
                logger.info(f"img0 size: {img0.shape}")
        if log_time:
            time_logger.stop("Pre Processing")
        # if len(img.shape) == 3:
        #     img = img.unsqueeze(0)
        if i%2 == 0 and log_time:
            time_logger.start("Full Pipeline")
        if i%2 == 1 and log_time and i != 1:
            time_logger.stop("Full Pipeline")

        if log_time:
            time_logger.start("Infrence")
        results = model.predict(img,
                                augment=args.augment,
                                verbose=False,
                                nms=True,
                                conf=args.conf_thres,
                                iou=args.iou_thres,
                                imgsz=img_shape)[0].cpu().numpy() # Inference
        pred = results.boxes.data
        pred[:,:4] = scale_boxes(img_shape, pred[:,:4], img0_shape).round()
        if log_time:
            time_logger.stop("Infrence")
        if args.disp_pred:
            disp_pred(pred,names_object,logger)
        if args.visualize:
            if log_time:
                time_logger.start("Visualize")
            # visualize_new(pred=results, img0=np.array(img), image_name="Schenk/Sign Predictions")
            visualize_yolo_2D(pred, img0=img0, img=img, args=args, names=names_object, line_thickness=3, rescale=False)#, classes_not_to_show=[0])
            if log_time:
                time_logger.stop("Visualize")

        if args.track:
            """
            If tracking is enabled, the tracker will be updated with the new predictions.
            """
            if log_time:
                time_logger.start("Tracking Frames")
            # Object Tracking
            best_frame = object_tracker.update(pred,img0,img)
            if log_time:
                time_logger.stop("Tracking Frames")
            if args.track_digits and (best_frame is not None or args.force_detect_digits):
                if log_time:
                    time_logger.start("Tracking Digit")
                # Digit Sequence Tracking.
                img0 = best_frame["image"] if best_frame is not None else img0
                sequence, valid, result_digit, pred_digit, img = dd.detect(img0=best_frame["image"]) if not args.force_detect_digits or best_frame is not None else dd.detect(img0)
                if args.visualize:
                    # visualize_new(pred=result_digit,img0=np.array(img),image_name="Digit Detector")
                    visualize_yolo_2D(pred_digit, img0=img0, img=img, args=args, names=names_digit, line_thickness=3, image_name="Digit Predictions", rescale=True)
                if log_time:
                    time_logger.stop("Tracking Digit")
                if valid:
                    logger.info(f"Predicted Sequence: {Fore.GREEN}{sequence}{Style.RESET_ALL}\n")
                    # TODO: Save to file or transmit to server. Maybe??
        if args.disp_pred or args.verbose:
            print("\n")
        if init:
            init = False
        if log_time:
            time_logger.stop("Internal Pipeline")
        log_time = args.log_time
        if (time.monotonic()-start_stream) > args.time and args.time != -1:
            if args.prog_bar:
                pbar.n = pbar.total
                pbar.close()
            break
        # if args.visualize:
        #     if log_time:
        #         time_logger.start("GUI-Visualize")
        #     gui.update_image(img1,0,0)
        #     if args.track_digits and best_frame is not None:
        #         gui.update_image(img2,0,1)
        #     gui.root.update()
        #     if log_time:
        #         time_logger.stop("GUI-Visualize")
        if args.wait:
            time.sleep(0.25)
            wait_for_input(live=live,args=args)


    if args.transmit:
        transmitter.stop_transmit_udp()
        transmitter.stop_transmit_ml()
    if log_time:
        time_logger.summarize()
    logger.info("Stream Done")

if __name__ == '__main__':
    main()
    #main_clean()

