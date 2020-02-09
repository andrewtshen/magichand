import sys; sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

# from .detect import determineCentroid
import detect
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imwrite("frame.jpg", frame)
    print("reading frame")

    cX, cY = detect.determineCentroid(frame)
    print("centroid: ", cX, cY)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()