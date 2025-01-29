from ultralytics import YOLO
import cv2
import os

imgTest = "D:/1_MAIN PROJECT/Image Processing/MRI_detection/test/images/Tr-gl_0163_jpg.rf.744856a107adc6196834a58021e16ca4.jpg"

img = cv2.imread(imgTest)
H, W, _ = img.shape

imgPredict = img.copy()

model_path = "D:/1_MAIN PROJECT/Image Processing Models/MRI model/weights/best.pt"

model = YOLO(model_path)

threshold = 0.5

results = model(imgPredict)[0]

for result in results.boxes.data.tolist():
    x1,y1,x2,y2,score,class_id = result

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)    
    y2 = int(y2)

    if score > threshold:
        cv2.rectangle(imgPredict, (x1, y1), (x2, y2), (0, 0, 255), 1)
        class_name = results.names[int(class_id)]
        cv2.putText(imgPredict, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



cv2.imshow("Prediction", imgPredict)
cv2.imshow("Original", img)
cv2.waitKey(0)