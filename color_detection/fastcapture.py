import sys; sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

# import the necessary packages
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import detect
import controlmouse
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
    help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
    help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# created a *threaded* video stream, allow the camera sensor to warmup, and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

click_check = [50000, 50000, 50000, 50000, 50000]

cntr = 0

frame = vs.read()
detect.initBounds(frame)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
