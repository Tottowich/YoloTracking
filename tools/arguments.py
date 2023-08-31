import argparse
from .utils import check_and_create_dir
from random_word import RandomWords
RANDOM_NAME = RandomWords()
import sys
from  pathlib import Path
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) # Add ROOT
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative Path
import yaml
def parse_config():
    """
    Parse the configuration file.
    """
    parser = argparse.ArgumentParser(description='arg parser')
    #parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
    #                    help='specify the config for demo')
    parser.add_argument('--weights', type=str, default=ROOT / './TrainedModels/Object/object.onnx', help='model path(s) detect objects')

    parser.add_argument('--source', type=str, default=None, help='model path(s)')

    parser.add_argument('--ip', type=str, default=None, help='ip address')
    parser.add_argument('--port', type=int, default=None, help='port')

    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=448, help='inference size h,w')
    parser.add_argument('--data', type=str, default=ROOT / "./TrainedModels/Object/data.yaml", help='(optional) dataset.yaml path')
    parser.add_argument('--max_det', type=int, default=10, help='maximum detections per image')
    parser.add_argument('--conf_thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--line_thickness', default=3, type=int, help='bounding box thickness (pixels) visualizations')
    parser.add_argument('--hide_labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide_conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference instead of FP32 (default)')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    
    parser.add_argument('--auto', action='store_true', help='auto size using the model')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    # parser.add_argument('--img_size', nargs='+', type=int, help='Size of input image')
    parser.add_argument('--name_run', type=str, default=f"{RANDOM_NAME.get_random_word()}_{RANDOM_NAME.get_random_word()}", help='specify the name of the run to save the results')

    # Object Tracking
    parser.add_argument('--object_frames', type=int, default=3, help='Take the last n frames to track the certainty of the prediction.')
    parser.add_argument('--tracker_thresh', type=float, default=0.6, help='Tracker threshold')
    parser.add_argument('--class_to_track', type=int, default=1, help='Class index to track')
    # Digit Tracking:
    parser.add_argument('--track_digits', action='store_true', help='Track the digits')
    parser.add_argument('--digit_frames', type=int, default=3, help='Take the last n frames to track the certainty of the prediction.')
    parser.add_argument("--weights_digits",type=str,default="./TrainedModels/Digit/digit.onnx",help="Path to model for digit detection")
    parser.add_argument("--conf_digits",type=float,default=0.3,help="Confidence threshold for digit detection")
    parser.add_argument("--iou_digits",type=float,default=0.2,help="NMS IoU threshold for digit detections")
    parser.add_argument("--ind_thresh",type=float,default=0.1,help="Individual threshold if a score of an individual digit is below this threshold then the sequence is invalid")
    parser.add_argument("--seq_thresh",type=float,default=0.2,help="Sequence threshold if the average score of the sequence is below this threshold then the sequence is invalid")
    parser.add_argument("--out_thresh",type=float,default=0.35,help="Output threshold if the average score of the sequence of sequences is below this threshold then the sequence history is invalid")
    parser.add_argument("--verbose",action="store_true",help="Whether to print information")
    parser.add_argument('--data_digit', type=str, default='./TrainedModels/Digit/data.yaml', help='(optional) dataset.yaml path to digit dataset.')
    parser.add_argument('--imgsz_digit', nargs='+', type=int, default=448, help='inference size h,w')
    parser.add_argument('--combination_file', type=str, default='./TrainedModels/data/combinations.txt', help='(optional) combination.txt path text file with currently valid digit combinations.')
    parser.add_argument('--time', type=int, default=-1
    , help='specify the time to stream data from a sensor')
    parser.add_argument('--output_folder', type=str, default=None, help='specify the output folder to save the results')
    parser.add_argument('--vid_stride', type=int, default=1, help='specify the stride of the video, 1 is every frame, 2 is every other frame, etc.')
    # Webcam should be a string which is the camera index. It should be activated as a boolean.
    parser.add_argument('--webcam', type=str, default=None, help='specify the webcam index')
    if sys.version_info >= (3,9):
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--visualize', action=argparse.BooleanOptionalAction)
        parser.add_argument('--wait', action=argparse.BooleanOptionalAction,help="Wait for keypress after each visualization")
        parser.add_argument('--prog_bar', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_time_log', action=argparse.BooleanOptionalAction)
        parser.add_argument('--force_detect_digits', action=argparse.BooleanOptionalAction)
        parser.add_argument('--track', action=argparse.BooleanOptionalAction)
        parser.add_argument('--save_csv', action=argparse.BooleanOptionalAction)
        parser.add_argument('--log_time', action=argparse.BooleanOptionalAction)
        parser.add_argument('--disp_pred', action=argparse.BooleanOptionalAction)
        parser.add_argument('--disp_time', action=argparse.BooleanOptionalAction)
        parser.add_argument('--transmit', action=argparse.BooleanOptionalAction)
        parser.add_argument('--log_all', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if isinstance(args.imgsz, list) and len(args.imgsz) == 1:
        args.imgsz = args.imgsz[0]
    if isinstance(args.imgsz,int):
        args.imgsz = (args.imgsz, args.imgsz)
    if isinstance(args.imgsz_digit, list) and len(args.imgsz_digit) == 1:
        args.imgsz_digit = args.imgsz_digit[0]
    if isinstance(args.imgsz_digit,int):
        args.imgsz_digit = (args.imgsz_digit, args.imgsz_digit)  
    data = {
        "object": read_yaml(args.data),
        "digit": read_yaml(args.data_digit)
    }
    if args.output_folder:
        check_and_create_dir(args.output_folder)
    

    return args, data
def read_yaml(path):
     with open(path,'r') as f:
        try:
            return yaml.safe_load(f)
        except:
            raise ValueError(f"Invalid data config file: {path}")