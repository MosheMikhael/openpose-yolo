# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:41:21 2018

@author: moshe.f
"""
import sys
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import draw
import neural_network as nn
import openpose_to_Json as op

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

#adr = 'input/run.mp4'
#adr = 'input/Russian Car Crash - Dash Cam Compilation.mp4'
#adr = 'input/IMG_8204.mp4'
#cam = cv2.VideoCapture(adr)
#width    = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
#height   = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
times = []
times_yolo = []
times_openpose = []
files = os.listdir('./input/Cars_driver_side')

fr_count = len(files)
#sys.exit()
#fr_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
#fourcc    = cv2.VideoWriter_fourcc(*'XVID')
#out_video = cv2.VideoWriter('output.avi',fourcc, 20.0, (width * 3, height))
#for itt in range(4000):
for itt in range(7, len(files)):
    print('START itt =', itt, 'from', fr_count)
    T = time.time()
#    _, image = cam.read()
    image = cv2.imread('./input/Cars_driver_side/' + files[itt])
    
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(g, (3, 3), 0)
    wide  = cv2.cvtColor(cv2.Canny(blurred, 0, 25), cv2.COLOR_GRAY2BGR)
    tight = cv2.cvtColor(cv2.Canny(blurred, 0, 10), cv2.COLOR_GRAY2BGR)
    auto  = cv2.cvtColor(auto_canny(blurred), cv2.COLOR_GRAY2BGR)
    width    = image.shape[1]
    height   = image.shape[0]
    
    t_y = time.time()
    output_yolo  = nn.get_features(image)
    times_yolo.append(time.time() - t_y)
    t_op = time.time()
#    output_opnps = op.getPose(image, True)
    pp, ll, output_opnps = op.getPose(image)
    times_openpose.append(time.time() - t_op)
    yolo_result = image.copy()
#    nose = image.copy()
    for box in output_yolo:
        name  = box[0]
        pt1   = box[1]
        pt2   = box[2]
        color = box[3]
        if name in ['person', 'car', 'truck']:
            yolo_result = draw.draw_bounding_box(yolo_result, name, color, pt1, pt2)
    yolo_result = op.drawPose(yolo_result, pp, ll, 3, 1)
    probMap = output_opnps[0, 0, :, :]
    probMap = cv2.resize(probMap, (width, height))
    minM = np.min(probMap)
    probMap -= minM
    maxM = np.max(probMap)
    k = 255 / maxM
    probMap *= 255
    probMap = np.uint8(probMap)
    out = np.hstack([yolo_result, wide, tight, auto])
#    pb = np.zeros((height, width, 3), np.uint8)
#    pb[:, :, 0] = probMap #blue
#    pb[:, :, 1] = probMap #green
#    pb[:, :, 2] = probMap #red
#    nose[:, :, 1] = 0
#    nose = cv2.addWeighted(nose, 0.3, pb, 1, 0)
#    out = np.zeros((height, 3 * width, 3), np.uint8)
#    out[:,       :  width, :] = image
#    out[:,  width: -width, :] = nose
#    out[:, -width:       , :] = yolo_result
#    out_video.write(out)
    
    cv2.imwrite('output_avi/Cars_driver_side/{}.png'.format(itt), out)
    T = time.time() - T
    times.append(T)
    Tl = (fr_count - itt) * sum(times) / len(times)
    Tl_h   = int(Tl // 3600) 
    Tl_min = int(Tl // 60 - 60 * Tl_h)
    Tl_sec = int(Tl % 60)
    print('END   itt =', itt, '\ttime loss:', Tl_h, 'h', Tl_min, 'min', Tl_sec, 'sec','\tcurrent itt:', T, 'sec')
#    break
print('avr time     yolo:', sum(times_yolo    ) / len(times_yolo    ))
print('avr time openpose:', sum(times_openpose) / len(times_openpose))
#cam.release()
#out_video.release()
