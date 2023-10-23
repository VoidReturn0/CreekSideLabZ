# Import necessary packages
import os
import cv2
import argparse
import numpy as np
import importlib.util
import time
from threading import Thread
from PIL import Image, ImageTk
import logging

logging.basicConfig(level=logging.DEBUG)

# Constants
rotation_angle = 270  # Camera rotation offset

lock_target = False  # or some default value as per your logic


class VideoStream:
    """Camera object that controls video streaming."""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(args.camindex)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
        self.stopped = False
        (self.grabbed, self.frame) = self.stream.read()

    def start(self):
        Thread(target=self.update).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


 

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', default='models/RivCam_lite')
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.95, type=float)
parser.add_argument('--resolution', default='640x480')
parser.add_argument('--edgetpu', action='store_true')
parser.add_argument('--camindex', default=0, type=int)
#parser.add_argument('--port', default=12345, type=int)  # Change 12345 to 12346 or another unused port
args = parser.parse_args()
#print('cam_args:',args)

CAM_INDEX = args.camindex
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu



# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()


# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#print(establish_connection)
#conn = establish_connection()

current_target_idx = 0


#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    
    

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    if frame1 is not None:
        frame = frame1.copy()
    else:
        print("Failed to capture frame!")
        continue  # or handle the error as you see fit


    # Acquire frame and resize to expected shape [1xHxWx3]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    imH, imW = frame.shape[:2]  # Frame dimensions
    center_x, center_y = imW // 2, imH // 2
    

    #Draw center crosshair
    crosshair_color = (0, 0, 255)
    crosshair_size = 15
    cv2.line(frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), crosshair_color, 2)
    cv2.line(frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), crosshair_color, 2)

    valid_detections = []


    if current_target_idx >= len(valid_detections):
        current_target_idx = 0


  
    # Before the loop over valid_detections
    if not lock_target or not primary_target:
        primary_target = valid_detections[0] if valid_detections else None
        lock_target = True

    # 1. Collect all valid detections
    valid_detections = []
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin, xmin, ymax, xmax = [int(dim) for dim in (boxes[i] * [imH, imW, imH, imW])]
            object_name = labels[int(classes[i])]
            if object_name != 'Index Head Rivet':
                continue
            x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
            x_offset, y_offset = x_center - center_x, y_center - center_y
            valid_detections.append((x_offset, y_offset, scores[i], xmin, ymin, xmax, ymax))

    # 2. Rank detections by confidence score
    valid_detections.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence score in descending order

    # 3. Draw bounding boxes and send offsets
    primary_target_identified = False
    for i, det in enumerate(valid_detections):
        x_offset, y_offset, _, xmin, ymin, xmax, ymax = det

        # If primary target not yet identified
        if not primary_target_identified and i == current_target_idx:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  # RED for primary target
            try:
                # Sending primary target offsets
                print('Needs x y offest message logic')
                
            except Exception as e:
                print(f"Failed to send message: {e}")
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

     # Rotate the frame just before displaying
    M = cv2.getRotationMatrix2D((imW/2, imH/2), rotation_angle, 1)
    rotated_frame = cv2.warpAffine(frame, M, (imW, imH))

    cv2.imshow('Object detector', rotated_frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    
    


# Cleanup after loop exits
cv2.destroyAllWindows()
videostream.stop()


