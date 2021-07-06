# https://github.com/AlexeyAB/darknet#pre-trained-models
# mygreatlearning.com/blog/yolo-object-detection-using-opencv/
import time

import cv2 as cv
import numpy as np

kitchen = cv.VideoCapture("rtsp://cctv:foscam@192.168.1.181:8080/videoMain")
#kitchen = cv.VideoCapture("rtsp://cctv:f0scam@192.168.1.184:88/videoMain")

net = cv.dnn.readNetFromDarknet('../models/DarkNet/yolov4-tiny.cfg', '../models/DarkNet/yolov4-tiny.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# save all the names in file o the list classes
with open("../models/DarkNet/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if not kitchen.isOpened():
    print("cannot open camera")
    exit()

fCount=0
kRet, kDetObj = kitchen.read()
mask = 0
mask_inv = 0
while True:
    fCount += 1

    kRet, kFrame = kitchen.read()

    if not kRet:
        print("Cannot read frame")
        break

#    kFrame = cv.resize(kFrame, None, fx=0.5, fy=0.5)
    if not fCount%24:
        fCount = 0

        # dimension must be divisible by 32
        kBlob = cv.dnn.blobFromImage(kFrame, 1/512.0, (416, 256), swapRB=True, crop=False)

        t0 = time.time()
        net.setInput(kBlob)
        outputs = net.forward(ln)
        t = time.time() - t0


        class_ids = []
        confidences = []
        boxes = []

        for out in outputs:

            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    height, width, channels = kFrame.shape
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

            # We use NMS function in opencv to perform Non-maximum Suppression
            # we give it score threshold and nms threshold as arguments.
            indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            kDetObj[:,:,:] = 0

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv.rectangle(kDetObj, (x, y), (x + w, y + h), color, 2)

                    cv.putText(kDetObj, f"{label}:{confidences[i]:.2f}", (x, y + 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)
                    cv.putText(kDetObj, f"Time:{t:.2f}", (0, 130), cv.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)

        ret, mask = cv.threshold(cv.cvtColor(kDetObj, cv.COLOR_BGR2GRAY), 10, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)
        kDetObj = cv.bitwise_and(kDetObj, kDetObj, mask=mask)
        kFrame = cv.bitwise_and(kFrame, kFrame, mask=mask_inv)

    cv.imshow('Kitchen', cv.add(kFrame,kDetObj) )

    if cv.waitKey(1) == ord('q'):
        break

kitchen.release()
cv.destroyAllWindows()
