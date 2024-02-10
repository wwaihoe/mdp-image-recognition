import tensorflow as tf
import tflite_runtime.interpreter as tflite
import cv2
from PIL import Image
import numpy as np


classes = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", 
           "Alphabet_A", "Alphabet_B", "Alphabet_C", "Alphabet_D", "Alphabet_E", "Alphabet_F", "Alphabet_G", "Alphabet_H", "Alphabet_S", "Alphabet_T", "Alphabet_U", "Alphabet_v", "Alphabet_w", "Alphabet_x", "Alphabet_y", "Alphabet_z", 
           "up_arrow", "down_arrow", "right_arrow", "left_arrow", "Stop", "Bullseye"]

interpreter = tflite.Interpreter(model_path="./models/prod/tflite_effnetv2_ft.tflite")
interpreter.allocate_tensors()


def classify(path):
  image = resize(path, 128, 128)
  image = np.array(image)
  # Convert RGB to BGR
  bgr_image = image[:, :, ::-1].copy()
  gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
  cols, rows = gray.shape
  brightness = np.sum(bgr_image) / (255 * cols * rows)
  minimum_brightness = 0.8
  ratio = brightness / minimum_brightness

  if ratio < 1:
    bgr_image = cv2.convertScaleAbs(bgr_image, alpha = 1 / ratio, beta = 0)
    
  #convert from BGR to RGB
  rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

  rgb_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.float32)/255.0
  print(rgb_tensor)
  img_batch = np.expand_dims(rgb_tensor, axis=0)

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], img_batch)
  interpreter.invoke()

  logits = interpreter.get_tensor(output_details[0]['index'])
  pred = int(tf.math.argmax(logits, axis=1)[0])

  return classes[pred]



def resize(path, new_width, new_height):
  with Image.open(path) as im:
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
    return im_resized