from ultralytics import YOLO
import cv2
import math
import time

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("C:/Users/suprava chakma/Downloads/try code/weights/yolov8m.pt")

classNames = ['Bus Stop', 'Construction Ahead', 'Crossroads Ahead', 'Filling Station Ahead', 'Go Left', 'Go Right', 'Go Slow', 'Hospital Ahead', 'Left Turn', 'Market Ahead', 'Mosque Ahead', 'Narrow Bridge Ahead', 'Narrow Road Ahead', 'Narrow Road on Left Ahead', 'No Horns', 'No-Overtaking', 'Rail Crossing Ahead', 'Right Turn', 'Road On Left', 'Road On Right', 'School Ahead', 'Speed Breaker Ahead', 'Speed Limit 20kmh', 'Speed Limit 40kmh', 'Speed Limit 60kmh', 'Speed Limit 80kmh', 'Stop', 'Street Crossing Ahead']

prev_frame_time = 0
new_frame_time = 0

def draw_corner_rect(img, bbox, color, thickness, length):
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h

    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def put_text_rect(img, text, pos, scale, thickness, color_bg, color_text):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_w, text_h = text_size

    x, y = pos
    cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y + 5), color_bg, cv2.FILLED)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color_text, thickness)

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            draw_corner_rect(img, (x1, y1, w, h), (255, 0, 255), 3, 15)
            
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            put_text_rect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), 1, 1, (255, 0, 255), (255, 255, 255))

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("YOLO Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
