# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:56:37 2018

@author: moshe.f
"""

import cv2
import time
import numpy as np
from random import randint
import json

def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    #find the blobs
    _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output, mapIdx, frameWidth, frameHeight, detected_keypoints, POSE_PAIRS):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
#            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list, mapIdx, POSE_PAIRS):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def makeDictOfPosePoints(personwiseKeypoints, keypoints_list, keypointsMapping):
    peoples = []
    idx = 0
    for p_idx in range(personwiseKeypoints.shape[0]):
        pose = {'Nose': None, 'Neck': None, 'R-Sho': None, 'R-Elb': None, 'R-Wr': None, 'L-Sho': None, 'L-Elb': None, 'L-Wr': None, 'R-Hip': None, 'R-Knee': None, 'R-Ank': None, 'L-Hip': None, 'L-Knee': None, 'L-Ank': None, 'R-Eye': None, 'L-Eye': None, 'R-Ear': None, 'L-Ear': None }          
        for part_idx in range(personwiseKeypoints.shape[1]):
            point_idx = int(personwiseKeypoints[p_idx][part_idx])
            if point_idx != -1 and part_idx < 18:
                point = keypoints_list[point_idx][:2]
                point = [int(point[0]), int(point[1])]
#                print(part_idx)
                part  = keypointsMapping[part_idx]
                pose[part] = point
        peoples.append({'id': idx, 'pose': pose})
        idx += 1
    return peoples

def makeDictOfLinesPoints(personwiseKeypoints, keypoints_list, POSE_PAIRS):
    peoples = []
    idx = 0
    for i in range(17):
        lines = []
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            lines.append([(B[0], A[0]), (B[1], A[1])])
        peoples.append({'id': idx, 'lines':lines})
        idx += 1
    return peoples

def getPose(image):
    protoFile = "coco/pose_deploy_linevec.prototxt"
    weightsFile = "coco/pose_iter_440000.caffemodel"
    nPoints = 18
    # COCO Output Format
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

    POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                  [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                  [1,0], [0,14], [14,16], [0,15], [15,17],
                  [2,17], [5,16] ]
    # index of pafs correspoding to the POSE_PAIRS
    # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
              [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
              [47,48], [49,50], [53,54], [51,52], [55,56],
              [37,38], [45,46]]
    
    frameWidth  = image.shape[1]
    frameHeight = image.shape[0]
#    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
#    print("Time Taken in forward pass = {}".format(time.time() - t))

    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1
    detected_keypoints = []
    for part in range(nPoints): # итерируемся по частям тела
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        keypoints = getKeypoints(probMap, threshold)
#        print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
        detected_keypoints.append(keypoints_with_id)
        
#    frameClone = image.copy()
    valid_pairs, invalid_pairs = getValidPairs(output, mapIdx, frameWidth, frameHeight, detected_keypoints, POSE_PAIRS)
    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list, mapIdx, POSE_PAIRS) # 

#    for i in range(17):
#        for n in range(len(personwiseKeypoints)):
#            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
#            if -1 in index:
#                continue
#            B = np.int32(keypoints_list[index.astype(int), 0])
#            A = np.int32(keypoints_list[index.astype(int), 1])
##            print((B[0], A[0]), (B[1], A[1]))
#            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
#    cv2.imwrite('output/' + name_img, frameClone)
    peoples_pose_points = makeDictOfPosePoints(personwiseKeypoints, keypoints_list, keypointsMapping)
    peoples_pose_lines  = makeDictOfLinesPoints(personwiseKeypoints, keypoints_list, POSE_PAIRS)
    return peoples_pose_points, peoples_pose_lines, output

def drawPose(image, points, lines, radius=5, line_width=1):
    colors = [[0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
              [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
              [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]
    for lines_ in lines:
        ls = lines_['lines']
        for line in ls:
            ptA = line[0]
            ptB = line[1]   
            cv2.line(image, (ptA[0], ptA[1]), (ptB[0], ptB[1]), (255, 255, 255), 2 * line_width, cv2.LINE_AA)
            cv2.line(image, (ptA[0], ptA[1]), (ptB[0], ptB[1]), (  0,   0,   0),     line_width, cv2.LINE_AA)
    for pts in points:
        pose = pts['pose']
        i = 0
        for pt_name in pose:
            pt = pose[pt_name]
            if pt is not None:
                cv2.circle(image, (pt[0], pt[1]), radius, colors[i], -1, cv2.LINE_AA)
            i += 1
    return image
if __name__ == '__main__':
    times = []
    for i in range(1, 31):
        print(i)
        T = time.time()
        name_img = '{}.jpeg'.format(i)
        img_adress = 'input/i/' + name_img
        image = cv2.imread(img_adress)
        points, lines, _ = getPose(image)
        T = time.time() - T
        times.append(T)
        print(' ', T)
        image = drawPose(image, points, lines)
        cv2.imwrite('output/i/' + name_img, image)
    print('avr = ', sum(times) / len(times))