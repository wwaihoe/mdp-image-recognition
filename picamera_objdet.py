from picamera import PiCamera
from objectdetection_yolov8 import detect
from time import sleep
from datetime import datetime

camera = PiCamera()


camera.start_preview()
sleep(1)
now = datetime.now()
path = f"/camera/{now}.jpg"
camera.capture(path)
preds = detect(path)
print(preds)
camera.stop_preview()

