## Introduction
one-stage visual grounding; minial version
## preparation

* Python 3.5
* Pytorch 0.4 or higher
* Pytorch-Bert https://github.com/huggingface/pytorch-pretrained-BERT
* Yolov3.weights https://pjreddie.com/media/files/yolov3.weights
* Datasets


## Training
1. Dataset: place the soft link of dataset folder in code/../ln_data/DMS/...
	We follow dataset structure from https://github.com/BCV-Uniandes/DMS
2. train_yolo.py is the main training and validation file; check darknet.py and textcam_yolo.py
	for models. referit_loader.py is the used dataloader

This repo is partly built on the YOLOv3 implementation (https://github.com/eriklindernoren/PyTorch-YOLOv3) and the data loader implemented by DMS (https://github.com/BCV-Uniandes/DMS).