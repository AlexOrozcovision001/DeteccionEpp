import argparse, time
import cv2
from ultralytics import YOLO

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--video',required=True);p.add_argument('--conf',type=float,default=.35);a=p.parse_args()
 model=YOLO(a.model);cap=cv2.VideoCapture(a.video);n=0;t=time.perf_counter()
 while True:
  ok,frame=cap.read()
  if not ok:break
  model.predict(frame,conf=a.conf,imgsz=640,verbose=False);n+=1
 print(f'{n} frames, {n/(time.perf_counter()-t):.2f} FPS promedio')
if __name__=='__main__':main()
