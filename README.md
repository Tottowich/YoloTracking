# Live-Yolo
Detection algorithm with utilizes both tracking and validation to make accurate predictions.
## Input arguments:
<details style="font-size: 16px;">
<summary> Object detection </summary>
<br>

| Argument | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
|--weights|str|ROOT/'./TrainedModels/object/object.onnx'|Path to the object detection model's weights.|--weights ./path/to/weights.onnx|
|--data|str|ROOT/"./TrainedModels/Object/data.yaml"|Path to the dataset configuration file.|--data ./path/to/data.yaml|
|--max_det|int|1000|Maximum number of detections per image.|--max_det 500|
|--conf_thres|float|0.6|Confidence threshold for object detection.|--conf_thres 0.5|
|--iou_thres|float|0.1|Intersection over Union (IoU) threshold for NMS.|--iou_thres 0.2|
|--track|action: BooleanOptionalAction||Enable tracking.|--track|
|--imgsz/--img/--img-size|int/list[int]|448|Inference size (height and width) for the input image.|--imgsz 512/--img-size 640 480|

</br>
</details>

<details style="font-size: 16px;">
<summary> Digit detection </summary>
<br>

| Argument | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
|--track_digits|action: store_true||Enable digit tracking.|--track_digits|
|--digit_frames|int|3|Number of frames to track for digit certainty.|--digit_frames 5|
|--weights_digits|str|"./TrainedModels/digit/digit.onnx"|Path to the model for digit detection.|--weights_digits ./path/to/digit_model.onnx|
|--conf_digits|float|0.3|Confidence threshold for digit detection.|--conf_digits 0.5|
|--iou_digits|float|0.1|IoU threshold for digit detections.|--iou_digits 0.2|
|--ind_thresh|float|0.1|Individual threshold for digit sequences.|--ind_thresh 0.2|
|--seq_thresh|float|0.2|Sequence mean threshold for digit sequences.|--seq_thresh 0.3|
|--out_thresh|float|0.35|Output threshold for sequence mean history.|--out_thresh 0.4|
|--data_digit|str|"./TrainedModels/digit/data.yaml"|Path to the dataset configuration file for digit detection.|--data_digit ./path/to/digit_data.yaml|
|--imgsz_digit|int/list[int]|448|Inference size (height and width) for digit detection.|--imgsz_digit 512/--imgsz_digit 640 480|
|--combination_file|str|"./TrainedModels/data/combinations.txt"|Path to the combination file.||

<br>
</details>

<details style="font-size: 16px;">
<summary> Inference settings </summary>
<br>

| Argument | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
|--object_frames|int|3|Number of frames to track for object certainty.|--object_frames 5|
|--tracker_thresh|float|0.6|Tracker threshold for object tracking.|--tracker_thresh 0.5|
|--class_to_track|int|1|Class index to track.|--class_to_track 2|
|--augment|action: store_true||Augmented inference.|--augment|
|--agnostic-nms|action: store_true||Class-agnostic NMS.|--agnostic-nms|
|--half|action: store_true||Use FP16 (half-precision) inference.|--half|
|--device|str|'cuda:0'|Which device to run inference on, e.g. mps, cpu, cuda.|--device cuda:0|
|--ckpt|str|None|Path to the pretrained model checkpoint.|--ckpt ./path/to/checkpoint.pth|
|--auto|action: store_true||Auto-size using the model.|--auto|

<br>
</details>

<details style="font-size: 16px;">
<summary> Visualization </summary>
<br>

| Argument | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
|--visualize|action: BooleanOptionalAction||Enable visualization.|--visualize|
|--wait|action: BooleanOptionalAction|Help: Wait for keypress after each visualization|--wait|
|--prog_bar|action: BooleanOptionalAction||Enable progress bar.|--prog_bar|
|--hide_labels|action: store_true|False|Hide object labels in visualizations.|--hide_labels|
|--hide_conf|action: store_true|False|Hide object confidences in visualizations.|--hide_conf|
|--line_thickness|int|3|Thickness of bounding box lines for visualizations.|--line_thickness 2|

<br>
</details>

<details style="font-size: 16px;">
<summary> Logging </summary>
<br>

| Argument | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
|--verbose|action: store_true||Print information during execution.|--verbose|
|--save_time_log|action: BooleanOptionalAction||Save time log.|--save_time_log|
|--save_csv|action: BooleanOptionalAction||Save results as CSV.|--save_csv|
|--log_time|action: BooleanOptionalAction||Log time during execution.|--log_time|
|--disp_pred|action: BooleanOptionalAction||Display predictions.|--disp_pred|
|--disp_time|action: BooleanOptionalAction||Display execution time.|--disp_time|
|--log_all|action: BooleanOptionalAction||Log all information.|--log_all|

<br>
</details>


<details style="font-size: 16px;">
<summary> General </summary>
<br>

| Argument | Type | Default | Description | Example |
| --- | --- | --- | --- | --- |
|--ip|str|None|IP address.|--ip 192.168.0.1|
|--port|int|None|Port number.|--port 8080|
|--name_run|str|randomly generated names|Name of the run to save the results.|--name_run my_run|
|--transmit|action: BooleanOptionalAction||Transmit data.|--transmit|
|--webcam|str||Use webcam as input. Which webcam to use.|--webcam "1"|
|--classes|int/list[int]||Filter detections by class index.|--classes 0/--classes 0 2 3|
|--source|str|None|Path to the input source.|--source ./path/to/input.mp4|

<br>
</details>

### Example usage:
```bash
python live_yolo.py --webcam "0" --track --track_digits --visualize --time 60 --disp_pred --object_frames 10 --class_to_track 1 --verbose
```
