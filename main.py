import numpy as np
import string
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


def get_car(license_plate, vehicle_track_ids):

    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


model_car = YOLO("../Yolo-Weights/yolov8l.pt")
model_lic = YOLO('./best.pt')

cap = cv2.VideoCapture("./car.mp4")  # For Video
ret, img = cap.read()
H, W, _ = img.shape
out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'mpv4'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

mask = cv2.imread("mask.png")

# Tracking
tracker_car = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

## Coco Names
classNames = []
with open("coco.names", 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
vehicles = [2, 3, 5, 7]

while ret:

    imgRegion = cv2.bitwise_and(img, mask)

    results_car = model_car(imgRegion)[0]

 
    detections = np.empty((0, 5))
    for detect in results_car.boxes.data.tolist():
        x1, y1, x2, y2, score, id = detect
        conf = math.ceil(score * 100) / 100

        if int(id) in vehicles and conf > 0.5:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    resultsTracker = tracker_car.update(detections)

    # detect license plates
    results_lic = model_lic(imgRegion)[0]
    for license_plate in results_lic.boxes.data.tolist():
        x1, y1, x2, y2, score, id = license_plate
        conf = math.ceil(score * 100) / 100

        if conf > 0.5:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=2, colorC=(255,0,255), colorR=(0, 0, 0))
            
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, resultsTracker)
            xcar1, ycar1, xcar2, ycar2 = int(xcar1), int(ycar1), int(xcar2), int(ycar2)

            wcar, hcar = xcar2 - xcar1, ycar2 - ycar1
            cvzone.cornerRect(img, (xcar1, ycar1, wcar, hcar), l=20, t=4,colorR=(0,0,0))


    #cv2.imshow("Image", re_img)
    #cv2.waitKey(1)
    out.write(img)
    ret, img = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

