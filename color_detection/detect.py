import sys; sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")

# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils

def initBounds(image):
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # keep max contour
    areas = [cv2.contourArea(c) for c in cnts]
    if len(areas) == 0:
        return (0, 0, 100000)
    max_idx = areas.index(max(areas))
    max_area = areas[max_idx]
    M = cv2.moments(cnts[max_idx])
    if M["m00"] == 0:
        return (0, 0, max_area)
    cX = int(M["m10"] / (M["m00"] + 0.1))
    cY = int(M["m01"] / (M["m00"] + 0.1))
    print ("DEBUG shape: ", image.shape)


def determineCentroid(image):
    # create NumPy arrays from the boundaries #BGR order
    lower = np.array([180, 0, 20], dtype = "uint8")
    upper = np.array([220, 100, 100], dtype = "uint8")
        

    # find the colors within the specified boundaries
    mask = cv2.inRange(image, lower, upper)

    # preprocess the data/slows it down
    # blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    # thresh = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # keep max contour
    areas = [cv2.contourArea(c) for c in cnts]
    if len(areas) == 0:
        return (0, 0, 100000)
    max_idx = areas.index(max(areas))
    max_area = areas[max_idx]

    M = cv2.moments(cnts[max_idx])
    if M["m00"] == 0:
        return (0, 0, max_area)
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

    cv2.imwrite("output.jpg", output)

    return (cX, cY, max_area)

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())
# # load the image
# image = cv2.imread(args["image"])
# determineCentroid(image)
