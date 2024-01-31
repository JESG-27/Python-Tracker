import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Load template from frame
template = cv.imread('source/template.jpg', 0)

# Load video for tracking
cap = cv.VideoCapture('source/sample_video.mp4')
frame_rate = int(cap.get(cv.CAP_PROP_FPS))
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Extract example frames for matching testing
example_frames = 3
matching_frames = []
for i in range(total_frames//example_frames, total_frames, total_frames//example_frames):
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    res, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    matching_frames.append(frame)

# Output video
output = cv.VideoWriter("output.avi", cv.VideoWriter_fourcc(*'MJPG'), 30, (int(cap.get(3)), int(cap.get(4)))) 

# Template matching method selection
methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
methods_results = []
height, width = template.shape
for method in methods:
    for frame in matching_frames:
        frame_copy = frame.copy()
        result = cv.matchTemplate(frame_copy, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc
        
        bottom_right = (location[0]+width, location[1]+height)
        cv.rectangle(frame_copy, location, bottom_right, 255, 2)

        methods_results.append(frame_copy)

#Display template matching results
fig, axs = plt.subplots(nrows=example_frames, ncols=len(methods))
index = 0
for i in range (0, len(methods)):
    for j in range (0, example_frames):
        axs[j,i].imshow(methods_results[index], cmap='gray')
        axs[j,i].get_xaxis().set_ticks([])
        axs[j,i].get_yaxis().set_ticks([])
        index += 1

axs[0,0].set_title("0) TM_CCOEFF")
axs[0,1].set_title("1) TM_CCOEFF_NORMED")
axs[0,2].set_title("2) TM_CCORR")
axs[0,3].set_title("3) TM_CCORR_NORMED")
axs[0,4].set_title("4) TM_SQDIFF")
axs[0,5].set_title("5) TM_SQDIFF_NORMED")

# Select best template matching method
plt.show(block=False)
while(True):
    option = input(f"Select the matching method (Number): ")
    try: 
        option = int(option)
        if option >= 0 and option <= 5:
            break
    except:
        print("Incorrect input")
        os.system("pause")
        os.system("cls")
plt.close()


# Video Tracking
print("Video Processing...")
trajectory = []
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
while (cap.isOpened()):
    ret, frame = cap.read()

    if (ret):
        grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        result = cv.matchTemplate(grey_frame, template, methods[option])
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if methods[option] in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        bottom_right = (location[0]+width, location[1]+height)
        cv.rectangle(frame, location, bottom_right, (0,0,255), 2)

        center = (location[0]+(width//2), location[1]+(height//2))
        trajectory.append(center)
        for circle in trajectory:
            cv.circle(frame, circle, 5, (0, 0, 255), -1)

        output.write(frame)

        if (cv.waitKey(25) == ord('q')):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()

# Plot trajectory
print("Trajectory Processing...")
fig = plt.figure()
x = []
y = []
for position in trajectory:
    x.append(position[0])
    y.append(-position[1])

new_x = np.linspace(min(x), max(x), 100)
y_inter = np.interp(new_x, x, y, period=10000)

plt.scatter(x, y, color='blue', s=10, label="match")
plt.plot(new_x, y_inter, color='red', label="Interpolation")

plt.show(block=False)

# Speed
frame_dif = np.absolute(np.absolute(max(y))-np.absolute(min(y)))
time = (frame_dif*1)/frame_rate
distance = float(input("Estimated distance (meters): "))
speed = distance/time # speed = distance/time, this is an estimate distance
print(f"Time: {time:.2f} seconds")
print(f"Estimated distance: {distance:.2f} meters")
print(f"Average estimated speed: {speed:.2f} meters per second")