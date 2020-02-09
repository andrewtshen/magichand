import sys; sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

def hello():
    print("hello world")

def determineCentroid(image):
    # create NumPy arrays from the boundaries #BGR order
    lower = np.array([200, 100, 0], dtype = "uint8")
    upper = np.array([255, 205, 100], dtype = "uint8")

    # find the colors within the specified boundaries
    mask = cv2.inRange(image, lower, upper)

    # preprocess the data
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    thresh = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    areas = [cv2.contourArea(c) for c in cnts]
    print ("areas: ", areas)
    if len(areas) == 0:
        return (0, 0)
    max_idx = areas.index(max(areas))
    # print ("cnts: ", areas)
    print ("max_idx: ", max_idx)
    print ("max_value: ", areas[max_idx])

    M = cv2.moments(cnts[max_idx])
    if M["m00"] == 0:
        return (0, 0)
    cX = int(M["m10"] / (M["m00"] + 0.1))
    cY = int(M["m01"] / (M["m00"] + 0.1))
    # draw the contour and center of the shape on the image
    cv2.drawContours(mask, [cnts[max_idx]], -1, (0, 255, 0), 2)
    cv2.circle(mask, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # write the image and the mask
    cv2.imwrite("mask.jpg", mask)
    output = cv2.bitwise_and(image, image, mask = mask)

    cv2.imwrite("output.jpg", image)

    return (cX, cY)

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())
# # load the image
# image = cv2.imread(args["image"])
# determineCentroid(image)