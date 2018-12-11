# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:15:17 2018

@author: moshe.f
"""

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import draw
import neural_network as nn
import openpose_to_Json as op
#itt = 1
times = []
times_yolo = []
times_openpose = []
for itt in range(1, 15 + 1):
    print('start itt =', itt)
    T = time.time()
    image = cv2.imread('input/img/i/cars&humans/{}.jpg'.format(itt))
    frameWidth  = image.shape[1]
    frameHeight = image.shape[0]
    t_y = time.time()
    output_yolo  = nn.get_features(image)
    times_yolo.append(time.time() - t_y)
    t_op = time.time()
    output_opnps = op.getPose(image, True)
    pp, ll = op.getPose(image)
    times_openpose.append(time.time() - t_op)
    yolo_result = image.copy()
    for box in output_yolo:
        name  = box[0]
        pt1   = box[1]
        pt2   = box[2]
        color = box[3]
        yolo_result = draw.draw_bounding_box(yolo_result, name, color, pt1, pt2)
    yolo_result = op.drawPose(yolo_result, pp, ll, 3, 1)
    cv2.imwrite('output/cars&humans/{}_yolo.png'.format(itt), yolo_result)
    probMap = output_opnps[0, 0, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))
    minM = np.min(probMap)
    probMap -= minM
    maxM = np.max(probMap)
    k = 255 / maxM
    probMap *= 255
    probMap = np.uint8(probMap)
    pb = np.zeros((frameHeight, frameWidth, 3), np.uint8)
#    pb[:, :, 0] = probMap #blue
    pb[:, :, 1] = probMap #green
#    pb[:, :, 2] = probMap #red
    image[:, :, 1] = 0
    out = cv2.addWeighted(image, 0.3, pb, 1, 0)
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.imshow(probMap, alpha=0.6)
    cv2.imwrite('output/cars&humans/{}_nose.png'.format(itt), out)
    T = time.time() - T
    times.append(T)
    print('end itt =', itt, '\ttime loss:', (30 - itt) * sum(times) / len(times), '\tcurrent itt:', T, 'sec')
print('avr time yolo:', sum(times_yolo) / len(times_yolo))
print('avr time yolo:', sum(times_openpose) / (2 * len(times_openpose)))