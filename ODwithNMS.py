import numpy as np
import cv2


thres = 0.5  # threshold to detect objects
nms_thres = 0.2 # no nms at 1

#img = cv2.imread('lena.jpg')

cap = cv2.VideoCapture(1)
#webcam width
cap.set(3, 640)
#webcam height
cap.set(4, 480)
#webcam brightness
cap.set(10,150)


classNames= []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print (classNames)


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
#Used model attributes
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold= thres)
    #changing bbox as single entity to list for the suppression of repeated bbox ID
    bbox = list(bbox)
    #Makes list of conf array, reshaped into 1D array (-1 for 1D, 1 for one row) (0 to remove outer bracket)
    confs = list(np.array(confs).reshape(1, -1)[0])
    #conf is a float32, mapping it to float
    confs = list(map(float, confs))
    print(classIds, bbox)

    #NMS: chooses the index of bbox thats repeated and then eliminates it
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_thres)


    for i in indices:
        # [0] removes outer bracket
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        #printing bounding box
        cv2.rectangle(img, (x,y), (x+w, h+y), color=(0, 255, 0), thickness=2)

        #printing class IDs (object names)
        cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # confidence text
        conftext = str(round(confs[i] * 100, 2)) +"%"

        cv2.putText(img, conftext, (box[0] + 200, box[1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # x, y text

        # Putting point in the center of the bounding box
        x2 = x + int(w / 2)
        y2 = y + int(h / 2)
        cv2.circle(img, (x2, y2), 4, (0, 255, 0), -1)

        # Printing the centroid coordinates point coordinates on the box
        text = "(x,y) = (" + str(x2) + "," + str(y2)+")"
        cv2.putText(img, text.upper(), (x2 - 10, y2 - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


    cv2.imshow("Output",img)
    cv2.waitKey(1)