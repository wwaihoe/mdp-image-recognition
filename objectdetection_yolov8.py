#pip3 install ultralytics

from ultralytics import YOLO
import cv2
import time
from PIL import Image
import numpy as np
from pathlib import Path

# Load the YOLOv8 model
model = YOLO("yolov8_ft.pt")
classes = model.names

def detect(path):
  # Preprocessing
  image_path = preprocess(path, 640, 640)
  # Perform object detection
  results = model(image_path)  
  # Visualize results 
  cv2.imshow("YOLOv8 Detection", results[0].plot())  
  filename = path[path.rindex('/')+1:]
  results.save(f"/results/{filename}")
  preds = []
  for c in results.boxes.cls:
    preds.append(classes[int(c)])
  return preds
   

def preprocess(path, new_width, new_height):
  with Image.open(path) as im:
    # Resize image
    width, height = im.size
    if height >= width:
        left = 0
        top = (height-width)/2
        right = width
        bottom = width+(height-width)/2
    else:
        left = (width-height)/2
        top = 0
        right = height+(width-height)/2
        bottom = height
    im_cropped = im.crop((left, top, right, bottom))
    im_resized = im_cropped.resize((new_width, new_height), resample=Image.HAMMING)
    # Convert RGB to BGR
    bgr_image = im_resized[:, :, ::-1].copy()
    # Alter brightness
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    cols, rows = gray.shape
    brightness = np.sum(bgr_image) / (255 * cols * rows)
    minimum_brightness = 0.8
    ratio = brightness / minimum_brightness

    if ratio < 1:
      bgr_image = cv2.convertScaleAbs(bgr_image, alpha = 1 / ratio, beta = 0)
      
    #convert from BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    filename = path[path.rindex('-')+1:]
    image_path = f"/images/{filename}"
    cv2.imwrite(image_path, rgb_image)
    return image_path
  

  
