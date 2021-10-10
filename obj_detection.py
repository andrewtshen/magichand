import cv2
import matplotlib.pyplot as plt
import numpy as np
import time 
import statistics as stat

hand_hist = None
traverse_point = []
traverse_centroid=[]
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]


    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)


def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if far_point is not None:
	        if len(traverse_point) < 20:
	        	traverse_point.append(far_point)
	        	traverse_centroid.append(cnt_centroid)
	        else:
	            traverse_centroid.pop(0)
	            traverse_point.pop(0)
	            traverse_centroid.append(cnt_centroid)
	            traverse_point.append(far_point)
        	draw_circles(frame, traverse_point)


def main():
    
    global hand_hist
    global traverse_point
    global traverse_centroid

    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)    
    count=1
    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        time.sleep(0.01)
        _, frame = capture.read()

        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist)
            if count%100==0:
            	move=det_gesture(traverse_point,traverse_centroid)
            	print(move)

        else:
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

        count+=1

    cv2.destroyAllWindows()
    capture.release()

def motion(traverse_point,traverse_centroid):
	
	if len(traverse_point)>0 and len(traverse_centroid)>0:
			
		#arrays containing the x-coordinates and y-coordinates of all the points in traverse_point
		
		p_x=np.array([i[0] for i in traverse_point])
		p_y=np.array([i[1] for i in traverse_point])
		
		#arrays containing the x_coordibares and y-coordinates of all the points in traverse_centroid
		
		c_x=np.array([i[0] for i in traverse_centroid])
		c_y=np.array([i[1] for i in traverse_centroid])
		
		#perform linear fit for the furthest points and centroid
		
		A_p = np.vstack([p_x, np.ones(len(p_x))]).T
		m_p, b_p = np.linalg.lstsq(A_p, p_y, rcond=None)[0]
		
		#perform linear fit for the centroid
		
		A_c = np.vstack([c_x, np.ones(len(c_x))]).T
		m_c, b_c = np.linalg.lstsq(A_c,c_y,rcond=None)[0]

		return (m_p, m_c)
	
	else:
		return None

def box_change(p1,p2):
	if int(p1[0]/(640/3))==int(p2[0]/(640/3)) and int(p1[1]/(160))==int(p2[1]/160):
		return False
	else:
		return True

def det_gesture(traverse_point, traverse_centroid):
	'''
	traverse_point: list of points the finger passes through
	traverse_centroid: list of points the centroid passes through
	
	returns: set (int)  
	0: move (both centre and fingers move)
	1: scroll (centre stays put and you rotate fingers)
	2: click (stay stable)

	'''
	set = None

	slopes=motion(traverse_point,traverse_centroid)

	if slopes is not None:
		m_p, m_c = slopes[0], slopes[1]

	change = box_change(traverse_centroid[0], traverse_centroid[-1])

	# condition for simply moving the cursor
	if change:
		set = 0
	else:
		if box_change(traverse_point[0],traverse_point[-1]):
			set = 1
		else:
			set=2

	return set

