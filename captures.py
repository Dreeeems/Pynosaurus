import pyautogui
import keyboard
import time
import os

output_file = "images"
os.makedirs(output_file,exist_ok= True)

x,y,width,height = 654,306,598,176

capturing = False

image_count = 0

while True:

    if keyboard.is_pressed("s") and not capturing:
        capturing = True

    
    if keyboard.is_pressed("q") and capturing:
        capturing = False
        break

    if capturing:
        screenshoot = pyautogui.screenshot(region=(x,y,width,height))
        screenshoot.save(os.path.join(f"images_{image_count:04d}.png"))
        image_count +=1
        time.sleep(0.1)


