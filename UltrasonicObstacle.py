from gpiozero import DistanceSensor
from gpiozero import Robot
from time import sleep
from LaneDetection import detect_line_segments, average_slope_intercept, region_of_interest

# Get the distance to the object
def get_distance(trigger, echo):
 
    # Send out a 10 microsecond pulse (ping)
    # from the trasmitter (TRIG)
    trigger.on()
    time.sleep(0.00001)
    trigger.off()
 
    # Start timer as soon as the reflected sound
    # wave is "heard" by the receiver (echo)
    while echo.is_active == False:
        pulse_start = time.time() # Time of last LOW reading
 
    # Stop the timer one the reflected sound wave
    # is done pushing through the receiver (ECHO)
    # Wave duration is proportional to duration of travel
    # of the original pulse.
    while echo.is_active == True:
        pulse_end = time.time() # Time of last HIGH reading
 
    pulse_duration = pulse_end - pulse_start
 
    # 34300 cm/s is the speed of sound
    distance = 34300 * (pulse_duration/2)
 
    # Round distance to two decimal places
    round_distance = round(distance,2)
 
    return(round_distance) 
    
 
while True:
    distance_to_object = get_distance(trigger,echo)
   
    # Avoid objects less than 15 cm away
    if distance_to_object <= 15:
        # determine movement here
        # use camera to get distance bt left edge of obstacle and lane and right edge of obstacle and lane
        # whichever is wider, set steering in the middle of those 2 edges, until obstacle no longer detected
        # reset steering to just whatever is in the middle of the 2 lane lines
        leftCropped = leftRegionInterest(leftGap())
        leftLineSegments = detect_line_segments(leftCropped)
        leftLines = average_slope_intercept(leftLineSegments)
        leftDist = leftLines[1][0] - leftLines[0][0]

        rightCropped = rightRegionInterest(rightGap())
        rightLineSegments = detect_line_segments(rightCropped)
        rightLines = average_slope_intercept(rightLineSegments)
        rightDist = rightLines[1][0] - rightLines[0][0]

        if leftDist < rightDist:
            # steer right, need to see pins and how pwm will affect, etc
        else:
            # steer left
        

    else:
        # normal lane detection and steering

def leftGap():
    lower_purple = np.array([255,20,130])
    upper_purple = np.array([255,20,150])
    maskPurple = cv2.inRange(img_hsv, lower_red, upper_red)

    lowerBlue = np.array([60, 40, 40])
    upperBlue = np.array([150, 255, 255])
    maskBlue = cv2.inRange(imgHsv, lowerBlue, upperBlue)

    maskPB = cv2.bitwise_or(maskPurple, maskBlue)
    edges = cv2.Canny(maskPB, 200, 400)

    return edges

def rightGap():
    lower_purple = np.array([255,20,130])
    upper_purple = np.array([255,20,150])
    maskPurple = cv2.inRange(img_hsv, lower_red, upper_red)

    lowerYellow = np.array([20, 50, 50])
    upperYellow = np.array([30, 255, 255])
    maskYellow = cv2.inRange(imgHsv, lowerYellow, upperYellow)

    return edges

def leftRegionInterest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    #isolate left-bottom half of screen
    polygon = np.array([[(0, height * 1 / 2), (width / 2, height * 1 / 2), (width / 2, height), (0, height)]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

def rightRegionInterest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    #isolate left-bottom half of screen
    polygon = np.array([[(width / 2, height * 1 / 2), (width, height * 1 / 2), (width, height), (width / 2, height)]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges
