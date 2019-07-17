from collections import deque
import numpy as np
import argparse
import time
import imutils
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
        help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
        help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "red"
#ClrLower = (166,84,141)
#ClrUpper = (186,255,255)
# ball in the HSV color space
#-----------for blue
ClrLower = (97, 100, 117)
ClrUpper = (117,255,255)
#-----------for green-----
#ClrLower = (29, 86, 6)
#ClrUpper = (64, 255, 255)
cap=cv2.VideoCapture(0)
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""
board = cv2.imread("board.jpg")
# keep looping
inline=None
clr=(0,0,0)
while True:
        __,frame = cap.read()
        frame=cv2.flip(frame,1)
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "red", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, ClrLower, ClrUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(frame, (int(x), int(y)), int(radius),
                                (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        pts.appendleft(center)
                        if(inline == None):
                                inline = center
                        else:
                                cv2.line(board,inline,center,clr,5)
                                inline = center

                else:
                        inline=None
        # show the frame to our screen and increment the frame counter
        cv2.imshow("Frame", board)
        cv2.imshow("Frame2", frame)
        cv2.imshow("hsv",hsv)
        cv2.imshow("mask",mask)
        
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        # press and hold b to not to draw
        # click s to save drawn image and to send to tesseract
        
        if key == ord("b"):
                clr = (255,255,255)
                inline = None
        if key == ord("c"):
                board = cv2.imread("board.jpg")
        else:
                clr = (0,0,0)
        if key == ord("q"):
                break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
