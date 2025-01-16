from ultralytics import YOLO
import cv2
 
model = YOLO('C:/Users/suprava chakma/Downloads/try code/weights/yolov8.pt')
results = model("images/2.jpg", show=True)
cv2.waitKey(0)