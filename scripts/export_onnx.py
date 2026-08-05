import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser(description='Exporta un modelo Ultralytics .pt para ONNX Runtime Web')
    p.add_argument('--weights',required=True)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--half',action='store_true')
    p.add_argument('--nms',action='store_true',help='Incorpora NMS en la salida si la versión de Ultralytics lo admite')
    a=p.parse_args()
    model=YOLO(a.weights)
    result=model.export(format='onnx',imgsz=a.imgsz,batch=1,dynamic=False,simplify=True,opset=17,half=a.half,nms=a.nms)
    print(f'Modelo exportado: {Path(result).resolve()}')
if __name__=='__main__':main()
