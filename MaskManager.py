# Main File
from picamera import PiCamera
from IMU import main
import time

def identify_mask():
    #TODO: @Shinjini
    return False

def warn_user():
    #TODO: Pawan
    pass

while True:
    image_file = "capture.png"
    camera = PiCamera()
    camera.capture(image_file)
    door_moving = main.get_movement()
    mask_on = identify_mask()
    if not mask_on and door_moving:
        warn_user()
    print(door_moving)
    time.sleep(0.3)



