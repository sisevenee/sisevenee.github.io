from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train():
    # model = YOLO("./ultralytics/cfg/models/v8/mtyolov8_myCBAM.yaml")
    # #model = YOLO("./yolov8n.pt")#系统的预训练权重
    # model.train(data="./ultralytics/cfg/datasets/mt.yaml", epochs=20)
    # result = model.val() 
    model = YOLO("./ultralytics/cfg/models/v8/oodyolov8.yaml")
    model.train(data="./ultralytics/cfg/datasets/ood.yaml", epochs=10)
    result = model.val() 


if __name__ == '__main__':
    train()