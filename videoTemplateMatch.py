import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the images in grey scale 
vid = cv2.VideoCapture('source/sample_video.mp4')
template = cv2.imread('source/ball_template.jpg', 0)
output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(vid.get(3)), int(vid.get(4)))) 

height, width = template.shape
trajectory = []

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
method = methods[3]

while (vid.isOpened()):
    ret, frame = vid.read()

    if (ret):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(grey_frame, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        bottom_right = (location[0]+width, location[1]+height)
        cv2.rectangle(frame, location, bottom_right, (0,0,255), 2)

        center = (location[0]+(width//2), location[1]+(height//2))
        trajectory.append(center)
        for circle in trajectory:
            cv2.circle(frame, circle, 5, (0, 0, 255), -1)

        output.write(frame)
        #cv2.imshow('frame', frame)

        if (cv2.waitKey(25) == ord('q')):
            break
    else:
        break

vid.release()
#cv2.destroyAllWindows()

# fig = plt.figure()
# for position in trajectory:
#     plt.scatter(position[0], -position[1], color='red')
# plt.show()