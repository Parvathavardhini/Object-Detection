import cv2
import numpy as np
from random import randrange

thres = 0.45 
nms_threshold = 0.2
video = cv2.VideoCapture(0)

Names= []
File = 'coco.names'
with open(File,'rt') as f:
    Names = f.read().rstrip('\n').split('\n')

configfile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsfile = 'frozen_inference_graph.pb'

frame = cv2.dnn_DetectionModel(weightsfile,configfile)
frame.setInputSize(320,320)
frame.setInputScale(1.0/ 127.5)
frame.setInputMean((127.5, 127.5, 127.5))
frame.setInputSwapRB(True)

while True:
    success,img = video.read()
    classIds, confs, bbox = frame.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y),(x+w,h+y),(randrange(256),randrange(256),randrange(256)), thickness=2)
        cv2.putText(img,Names[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    cv2.imshow("Output",img)
    key=cv2.waitKey(1)
    if key==81 or key==113:
        break

video.release()
print("Code Completed")