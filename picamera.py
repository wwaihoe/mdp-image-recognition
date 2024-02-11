from picamera import PiCamera
from classification import classify
from time import sleep

camera = PiCamera()

camera.start_preview()
sleep(5)
path = "/camera/cam_image.jpg"
camera.capture(path)
pred = classify(path)
print(pred)
camera.stop_preview()

