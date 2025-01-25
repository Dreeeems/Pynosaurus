# Check if the model works correctly

from ultralytics import YOLO


model = YOLO('runs/detect/train3/weights/best.pt')  


results = model.predict('url')  

results[0].show()