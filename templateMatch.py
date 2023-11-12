import numpy as np
import cv2

# Load the images in grey scale
img = cv2.imread('source/example_frame.jpg', 0)   
template = cv2.imread('source/ball_template.jpg', 0)

height, width = template.shape

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

for method in methods:
    img2 = img.copy()
    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    
    bottom_right = (location[0]+width, location[1]+height)
    cv2.rectangle(img2, location, bottom_right, 255, 2)
    
    down_width = 600
    down_height = 450
    down_points = (down_width, down_height)
    resized_down = cv2.resize(img2, down_points, interpolation= cv2.INTER_LINEAR)
    
    cv2.imshow('Match', resized_down)
    cv2.waitKey(0)
    cv2.destroyAllWindows()