import mss
import cv2
import numpy as np
import keyboard
import torch
from ultralytics import YOLO


model = YOLO('runs/detect/train3/weights/best.pt')  


def makemove(boxes):
    dinosaur = None
    closest_cactus = None
    min_distance = float('inf')  

    dinosaurs = []


    for box in boxes:
        confidence = box.conf[0].item() if isinstance(box.conf[0], torch.Tensor) else box.conf[0]
        if confidence >= 0.70:  #
            x1, y1, x2, y2 = (coord.item() if isinstance(coord, torch.Tensor) else coord for coord in box.xyxy[0])  
            class_id = int(box.cls[0].item()) if isinstance(box.cls[0], torch.Tensor) else int(box.cls[0])  
            label = model.names[class_id] 

            if label == "dinosaur":
                dinosaurs.append((x1, x2, y1, y2))  
            elif label == "cactus":
                if dinosaurs:  
                    if x2 > dinosaurs[0][0]:  
                        distance = x2 - dinosaurs[0][0]  
                        if distance < min_distance:  
                            closest_cactus = (x1, x2, y1, y2)
                            min_distance = distance
                            print(min_distance)
                            print(closest_cactus)

    if dinosaurs and closest_cactus:
        d_x1, d_x2, d_y1, d_y2 = dinosaurs[0]  
        c_x1, c_x2, c_y1, c_y2 = closest_cactus
        if c_x1 <= 130:  #
            print("Cactus proche ! Saut !")
            keyboard.press_and_release("space")


with mss.mss() as sct:
    region = {"top": 300, "left": 650, "width": 610, "height": 180}
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
            makemove(boxes)
                     
        if keyboard.is_pressed("q"):
            started = False
            break
