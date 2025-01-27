import mss
import cv2
import numpy as np
import keyboard
import torch
from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')


previous_cactus_x = None
previous_time = None


def makemove(boxes, current_time):
    global previous_cactus_x, previous_time

    dinosaur = None
    closest_cactus = None
    min_distance = float('inf')
    cactus_speed = 0
    potential_cacti = []

    for box in boxes:
        confidence = box.conf[0].item() if isinstance(box.conf[0], torch.Tensor) else box.conf[0]
        if confidence >= 0.70:
            x1, y1, x2, y2 = (coord.item() if isinstance(coord, torch.Tensor) else coord for coord in box.xyxy[0])
            class_id = int(box.cls[0].item()) if isinstance(box.cls[0], torch.Tensor) else int(box.cls[0])
            label = model.names[class_id]

            if label == "dinosaur":
                dinosaur = (x1, x2, y1, y2)
            elif label == "cactus" and dinosaur:
                if x2 > dinosaur[0]:  
                    distance = x2 - dinosaur[0]
                    potential_cacti.append((x1, x2, y1, y2, distance))


    if potential_cacti:

        potential_cacti = sorted(potential_cacti, key=lambda x: x[4])


        closest_cactus = potential_cacti[0]
        c_x1, c_x2, c_y1, c_y2, min_distance = closest_cactus


        if previous_cactus_x is not None and previous_time is not None:
            delta_time = current_time - previous_time
            if delta_time > 0:
                cactus_speed = (previous_cactus_x - c_x1) / delta_time


        previous_cactus_x = c_x1
        previous_time = current_time

        dynamic_factor = max(0.30, cactus_speed / 500)
        
        adjusted_threshold = min((210 * dynamic_factor), 100 + (cactus_speed * dynamic_factor))


        if min_distance <= adjusted_threshold:
            print("Jump")
            keyboard.press("space")


with mss.mss() as sct:
    region = {"top": 300, "left": 650, "width": 610, "height": 180}
    started = False
    while True:
        if keyboard.is_pressed("a"):
            started = True

        if started:
            screenshot = sct.grab(region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            results = model(frame)
            boxes = results[0].boxes

            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            makemove(boxes, current_time)

        if keyboard.is_pressed("q"):
            started = False
            break
