import sys; sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

# from .detect import determineCentroid
import detect
import numpy as np
import cv2
import controlmouse
from imutils.video import WebcamVideoStream
from imutils.video import FPS

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite("frame.jpg", frame)

    cX, cY, maxArea = detect.determineCentroid(frame)
    cmd = [0, cX, cY]
    controlmouse.execCmd(cmd)
    # print("centroid: ", cX, cY)
    print("area: ", maxArea)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()