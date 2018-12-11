# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:47:07 2018

@author: moshe.f
"""
import cv2
import numpy as np

class yoloNetWork:
    def __init__(self, version='3'):
        
        self.classes_adr = 'input/for yolo/classes.txt'
        self.v3_weights_adr = 'input/for yolo/yolov3.weights'
        self.v3_config_adr  = 'input/for yolo/yolov3.cfg'
        if version == 'tiny':
            self.v3_weights_adr = 'input/for yolo/yolov3-tiny.weights'
            self.v3_config_adr  = 'input/for yolo/yolov3-tiny.cfg'
        
        self.classes = None
        with open(self.classes_adr, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet(config=self.v3_config_adr, model=self.v3_weights_adr)
        

    
    def setInput(self, blob):
        self.net.setInput(blob)
        
    def forward(self, output_layers):
        return self.net.forward(output_layers)
    
    
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_features(img, net = yoloNetWork()):
    scale = 0.00392
    Width = img.shape[1]
    Height = img.shape[0]
    more = 5
    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net.net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out_data in outs:
        for detection in out_data:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # go through the detections remaining
    # after nms and draw bounding box
    boxes_out = []
    for ind in indices:
        i_ = ind[0]
        box = boxes[i_]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3] 
        #[{'className', [pt1], [pt2], [color], id}]
        boxes_out.append([net.classes[class_ids[i_]], [round(x - more * w / 100), round(y - more * h / 100)], [round(x + (1 + more / 100) * w), round(y + (1 + more / 100) * h)], net.COLORS[class_ids[i_]], class_ids[i_]])
    
    return boxes_out