import mss
import cv2
import numpy as np
import keyboard
from ultralytics import YOLO


model = YOLO('runs/detect/train3/weights/best.pt')  


with mss.mss() as sct:
    region = {"top": 300, "left": 650, "width": 680, "height": 180}
    started = False
    image_saved = False
    while True:
        if keyboard.is_pressed("a"):
            started  = True

        if started:
            
            screenshot = sct.grab(region)
            frame = np.array(screenshot)


            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if not image_saved:
                cv2.imwrite("captured_test_image.jpg", frame)
                print("Image sauvegardée sous le nom 'captured_test_image.jpg'")
                image_saved = True  

            results = model(frame)


            boxes = results[0].boxes

            for box in boxes:
             confidence = box.conf[0]  
             if confidence >= 0.80:  
                    x1, y1, x2, y2 = box.xyxy[0]  
                    class_id = int(box.cls[0])  
                    label = model.names[class_id]
                
                    print(f"Object - Label: {label}, Confidence: {confidence:.2f}, "
                      f"Coordinates: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")
                    
        if keyboard.is_pressed("q"):
            started = False
            break
