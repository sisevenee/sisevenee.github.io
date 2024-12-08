from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train():
    # model = YOLO("./ultralytics/cfg/models/v8/mtyolov8.yaml")
    model = YOLO("./yolov8n.pt")
    model.train(data="./ultralytics/cfg/datasets/mt.yaml", epochs=20)
    result = model.val()

if __name__ == '__main__':
    train()
