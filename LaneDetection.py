import cv2
import numpy as np

def getLaneLines(img): 
    # can use canny edges to get edges
    # or use colour, which is what is done here
    imgThres = thresholding(img)
    cropped_edges = region_of_interest(imgThres)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(img, line_segments)
    
    return lane_lines

def thresholding(img): # get yellow and blue edges
    # converts into hsv colour format
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerYellow = np.array([20, 100, 100], dtype = 'uint8')
    upperYellow = np.array([30, 255, 255], dtype = 'uint8')
    maskYellow = cv2.inRange(imgHsv, lowerYellow, upperYellow)

    lowerBlue = np.array([60, 40, 40])
    upperBlue = np.array([150, 255, 255])
    maskBlue = cv2.inRange(imgHsv, lowerBlue, upperBlue)

    maskYB = cv2.bitwise_or(maskYellow, maskBlue)
    edges = cv2.Canny(maskYB, 200, 400)

    return edges

def region_of_interest(edges): # isolate yellow and blue edges to lower half
    height, width = edges.shape
    mask = np.zeros_like(edges)

    #isolate bottom half of screen
    polygon = np.array([[(0, height * 1 / 2), (width, height * 1 / 2), (width, height), (0, height)]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1
    angle = np.pi /180
    min_threshold = 10
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=0, maxLineGap=4)

    return line_segments
    # printing this will show endpts x1,y1, x2,y2, and length of each line segment

def make_coordinate(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(3./5.)) # make points from 0.6 of frame down
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    # take each line, get bottom vertices, get slope and intercept, get avg, 
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinate(img, left_fit_average)
    right_line = make_coordinate(img, right_fit_average)
    return np.array([left_line, right_line]) # return left and right line as array

# plot lane lines on top of og img
def display_lines(img, lines, lineColour=(0,255,0), line_width=2):
    line_img = np.zeros_like(img)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            # takes coords of avg lines and draws them green on top of img, line width = 5
            cv2.line(line_img,(x1,y1),(x2,y2),lineColour,line_width)
    line_img = cv2.addWeighted(img, 0.8, line_img,1,1)
    return line_img


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        print("unable to readcamera feed")
    width = int(cap.get(3))
    height = int(cap.get(4))
    print('width is ' + str(width) + ' and height is ' + str(height))

    while True:

        success, img = cap.read()
        print('Original Dimensions : ', img.shape)
        img = cv2.resize(img,(480, 240), interpolation = cv2.INTER_AREA)
        lane_lines = getLaneLines(img)
        lane_lines_img = display_lines(img, lane_lines)
        cv2.imshow('lane lines', lane_lines_img)
        cv2.waitKey(1)