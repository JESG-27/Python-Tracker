import numpy as np
import cv2

# Load the images in grey scale 
vid = cv2.VideoCapture('source/sample_video.mp4')
template = cv2.imread('source/template_2.jpg', 0)

height, width = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

while (vid.isOpened()):
    ret, frame = vid.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (ret):
        result = cv2.matchTemplate(grey_frame, template, methods[3])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        location = max_loc

        bottom_right = (location[0]+width, location[1]+height)
        cv2.rectangle(frame, location, bottom_right, 255, 2)

        cv2.imshow('frame', frame)

        if (cv2.waitKey(25) == ord('q')):
            break
    else:
        break

vid.release()
cv2.destroyAllWindows()