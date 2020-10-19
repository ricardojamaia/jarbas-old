import sys
import os
import getopt
import cv2
import time
import datetime
import picar
from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo

from approxeng.input.selectbinder import ControllerResource

# Sample period for image capture capture one frame every IMG_SAMPLING_PERIOD seconds
IMG_SAMPLING_PERIOD = 1.0

# Configure maximum and minimum angle values for front wheels
FW_ANGLE_MAX = 90 + 30
FW_ANGLE_MIN = 90 - 30

# Configure maximum and minimum speed values
BW_SPEED_MIN = -100
BW_SPEED_MAX = 100

# Configure maximum and minimum values for camera pan and tilt
PAN_ANGLE_MAX = 170
PAN_ANGLE_MIN = 10
TILT_ANGLE_MAX = 150
TILT_ANGLE_MIN = 70

# configure maximum and minimum values provided by controller axis
AXIS_MAX = 1.0
AXIS_MIN = -1.0

# Calculate rates between axis and corresponding parameters
RATE_ANGLE = (FW_ANGLE_MAX - FW_ANGLE_MIN) / (AXIS_MAX - AXIS_MIN)
RATE_SPEED = (BW_SPEED_MAX - BW_SPEED_MIN) / (AXIS_MAX - AXIS_MIN)
RATE_PAN = (PAN_ANGLE_MAX - PAN_ANGLE_MIN) / (AXIS_MAX - AXIS_MIN)
RATE_TILT = (TILT_ANGLE_MAX - TILT_ANGLE_MIN) / (AXIS_MAX - AXIS_MIN)

# Set up PiCar
picar.setup()
bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)

# Offsets
fw.offset = 0
pan_servo.offset = 0
tilt_servo.offset = -50

# Set Motors and Servos to initial positions
bw.speed = 0
fw.turn(90)
pan_servo.write(90)
tilt_servo.write(90)

# Capture video
cap = cv2.VideoCapture(0)

# By default output pictures to current directory
OUTPUT_DIR = os.getcwd()

def process_argv(argv):
    output_dir = OUTPUT_DIR
    try:
        opts, args = getopt.getopt(argv[1:], "ho:", ["output="])
    except getopt.GetoptError:
        print('{0} -o <outputdir>'.format(argv[0]))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('{0} -o <outputdir>'.format(argv[0]))
            sys.exit()
        elif opt in ("-o", "--output"):
            output_dir = arg

    print('Saving pictures to {0}'.format(output_dir))

    return output_dir

def main():
    last_frame_time = 0

    while True:
        try:
            with ControllerResource() as gamepad:
                print('Found a gamepad and connected.')
                while gamepad.connected:

                    ctrl_axis_x, ctrl_axis_y = gamepad['r']
                    cam_axis_x, cam_axis_y = gamepad['l']
                    cam_axis_x, cam_axis_y = -cam_axis_x, -cam_axis_y
                    
                    angle = int((ctrl_axis_x - AXIS_MIN) * RATE_ANGLE + FW_ANGLE_MIN)
                    fw.turn(angle)
                    
                    speed = int((ctrl_axis_y - AXIS_MIN) * RATE_SPEED + BW_SPEED_MIN)

                    if speed > -25 and speed < 25:
                        bw.speed = 0
                        bw.stop()
                    elif speed >= 25:
                        bw.speed = abs(speed)
                        bw.forward()
                    else:
                        bw.speed = abs(speed)
                        bw.backward()
                    
                    cam_pan = int((cam_axis_x - AXIS_MIN) *
                                RATE_PAN + PAN_ANGLE_MIN)
                    pan_servo.write(cam_pan)
                    
                    cam_tilt = int((cam_axis_y - AXIS_MIN) *
                                RATE_TILT + TILT_ANGLE_MIN)
                    tilt_servo.write(cam_tilt)

                    # Capture images
                    cur_time = time.time()
                    _, frame = cap.read()
                    if cur_time - last_frame_time > IMG_SAMPLING_PERIOD:
                        last_frame_time = cur_time
                        
                        ts = datetime.datetime.fromtimestamp(
                            cur_time).strftime('%Y_%m_%d-%H_%M_%S')
                        #print('{0}/{1}.png'.format(OUTPUT_DIR, ts))
                        cv2.imwrite('{0}/{1}.png'.format(OUTPUT_DIR, ts), frame)


            # gamepad disconnected...
            print('Connection to gamepad lost')
        except IOError:
            # No gamepad found, wait for a bit before trying again
            print('Unable to find any gamepads')
            time.sleep(1.0)

def destroy():
    cv2.destroyAllWindows()
    bw.stop()

if __name__ == '__main__':
    try:
        OUTPUT_DIR = process_argv(sys.argv)
        main()
    except KeyboardInterrupt:
        print('Exiting ...')
        destroy()
