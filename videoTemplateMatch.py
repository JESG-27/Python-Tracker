import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the images in grey scale 
vid = cv2.VideoCapture('source/sample_video.mp4')
frame_rate = int(vid.get(cv2.CAP_PROP_FPS))
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
cv2.destroyAllWindows()

fig = plt.figure()
positions = [[],[]]
for position in trajectory:
    positions[0].append(position[0])
    positions[1].append(-position[1])

plt.scatter(positions[0], positions[1], color='red', s=10)

max = np.argmax(positions[1])
min = np.argmin(positions[1])

frame_dif = np.absolute(np.absolute(max)-np.absolute(min))
time = (frame_dif*1)/frame_rate
distance = float(input("Estimated drop height: "))
speed = distance/time # speed = distance/time, this is an estimate distance
print(f"Average speed: {speed:.2f} meters per second")

plt.show()