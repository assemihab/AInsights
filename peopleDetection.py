#################installations#################
# !pip install ultralytics -q
#################imports#################

import cv2
import pickle
from ultralytics import YOLO
from IPython.display import display, Image, clear_output,Video
import subprocess
import pandas as pd
import numpy as np
import os
#################paths and inits#################
path='/kaggle/input/barkotel-internship/AI Intern Video Tech Task.mp4'
yolopath='/kaggle/input/yolo-v8-vehicles-detecting-counting/yolov8x.pt'
# path='AI Intern Video Tech Task.mp4'
boxcolor=(0,255,0)
frame_width = 3
top_left=(100,100)
bottom_right=(300,360)
cls=[0]
skip_frames = 10
video=cv2.VideoCapture(path)
model = YOLO('yolov8x.pt')
# dict_classes = model.model.names
# print(dict_classes)

#geting video info
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video.get(cv2.CAP_PROP_FPS)
frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#output video
video_name = 'result.mp4'
output_path = "rep_" + video_name
tmp_output_path = "tmp_" + output_path
VIDEO_CODEC = "MP4V"

output_video = cv2.VideoWriter(tmp_output_path, 
                               cv2.VideoWriter_fourcc(*VIDEO_CODEC), 
                               fps, (width, height))

#################looping through the video#################
for i in range(frames):
    
    ret,frame=video.read()
    if not ret:
        break
    cv2.rectangle(frame,top_left,bottom_right,boxcolor,frame_width)
        
    y_hat = model.predict(frame, conf = 0.1, classes = cls, device = 0, verbose = False)
#     print("y hat is: ", y_hat)
    # boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
    conf    = y_hat[0].boxes.conf.cpu().numpy()
    # classes = y_hat[0].boxes.cls.cpu().numpy() 
#     print("classes boxs and conf: ",classes, boxes,conf)
    positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    count=0
    for ix, row in enumerate(positions_frame.iterrows()):
        # Getting the coordinates of each vehicle (row)
        xmin, ymin,xmax , ymax, _, _,  = row[1].astype('int')
        # print("the confidence is: ", confidence)
        if (xmax <= top_left[0] or xmin >= bottom_right[0] or
            ymax <= top_left[1] or ymin >= bottom_right[1]):
            # No intersection, draw the rectangle
            if conf[ix]>0.4:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(img=frame, text=str(np.round(conf[ix],2)),
                        org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
                count += 1
        else:
#             tracker.init(frame, bbox)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            cv2.putText(img=frame, text='Casheir_'+str(np.round(conf[ix],2)),
                    org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
            
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 5) # box
        
#         count=count+1
    cv2.putText(frame, f'People Count:{count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    output_video.write(frame)
    _, buffer = cv2.imencode('.jpg', frame)
    img = Image(data=buffer)

    # Clear the previous output and display the new frame
    clear_output(wait=True)
    display(img)
output_video.release()
if os.path.exists(output_path):
    os.remove(output_path)
subprocess.run(
    ["ffmpeg",  "-i", tmp_output_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_path])
os.remove(tmp_output_path)


