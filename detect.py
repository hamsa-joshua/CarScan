
# car_damage_detection.py

import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from scipy.spatial import distance

# === Set Paths ===
BASE_DIR = "path_to_unzipped_dataset/archive"  # Update this
IMG_DIR = os.path.join(BASE_DIR, "img")
VAL_JSON = os.path.join(BASE_DIR, "val", "COCO_val_annos.json")
MUL_VAL_JSON = os.path.join(BASE_DIR, "val", "COCO_mul_val_annos.json")

# === Register Datasets ===
register_coco_instances("damage_dataset", {}, VAL_JSON, IMG_DIR)
register_coco_instances("parts_dataset", {}, MUL_VAL_JSON, IMG_DIR)

# === Load Metadata ===
damage_metadata = MetadataCatalog.get("damage_dataset")
parts_metadata = MetadataCatalog.get("parts_dataset")

# === Setup Config for Damage Model ===
cfg_dmg = get_cfg()
cfg_dmg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_dmg.DATASETS.TRAIN = ("damage_dataset",)
cfg_dmg.DATASETS.TEST = ()
cfg_dmg.DATALOADER.NUM_WORKERS = 2
cfg_dmg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
cfg_dmg.SOLVER.IMS_PER_BATCH = 2
cfg_dmg.SOLVER.BASE_LR = 0.00025
cfg_dmg.SOLVER.MAX_ITER = 300  # increase for better results
cfg_dmg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg_dmg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # damage only

# === Train Damage Model ===
os.makedirs(cfg_dmg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg_dmg)
trainer.resume_or_load(resume=False)
trainer.train()

# === Setup Config for Parts Model ===
cfg_parts = get_cfg()
cfg_parts.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg_parts.DATASETS.TRAIN = ("parts_dataset",)
cfg_parts.DATASETS.TEST = ()
cfg_parts.DATALOADER.NUM_WORKERS = 2
cfg_parts.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
cfg_parts.SOLVER.IMS_PER_BATCH = 2
cfg_parts.SOLVER.BASE_LR = 0.00025
cfg_parts.SOLVER.MAX_ITER = 300
cfg_parts.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg_parts.MODEL.ROI_HEADS.NUM_CLASSES = 5  # headlamp, hood, rear bumper, front bumper, door

# === Train Parts Model ===
os.makedirs(cfg_parts.OUTPUT_DIR, exist_ok=True)
trainer2 = DefaultTrainer(cfg_parts)
trainer2.resume_or_load(resume=False)
trainer2.train()

# === Inference and Damage Matching ===
cfg_dmg.MODEL.WEIGHTS = os.path.join(cfg_dmg.OUTPUT_DIR, "model_final.pth")
cfg_dmg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
damage_predictor = DefaultPredictor(cfg_dmg)

cfg_parts.MODEL.WEIGHTS = os.path.join(cfg_parts.OUTPUT_DIR, "model_final.pth")
cfg_parts.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
parts_predictor = DefaultPredictor(cfg_parts)

def detect_damage_parts(img_path):
    image = cv2.imread(img_path)
    damage_outputs = damage_predictor(image)
    parts_outputs = parts_predictor(image)

    damage_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
    parts_boxes = parts_outputs["instances"].pred_boxes
    parts_classes = parts_outputs["instances"].pred_classes.tolist()
    parts_centers = parts_boxes.get_centers().tolist()

    parts_class_map = {0:'headlamp', 1:'rear_bumper', 2:'door', 3:'hood', 4:'front_bumper'}
    part_names = [parts_class_map[i] + f"_{j}" for j, i in enumerate(parts_classes)]

    part_dict = dict(zip(part_names, parts_centers))
    damage_dict = {f"damage_{i}": c for i, c in enumerate(damage_centers)}

    # Match based on distance
    result = {}
    for d_key, d_center in damage_dict.items():
        min_dist = float('inf')
        closest_part = None
        for p_key, p_center in part_dict.items():
            dist = distance.euclidean(d_center, p_center)
            if dist < min_dist:
                min_dist = dist
                closest_part = p_key
        result[d_key] = closest_part

    print("Damage Mapping:")
    for k, v in result.items():
        print(f"{k} => {v.split('_')[0]} (likely damaged)")

    return result

# Run on a sample image
detect_damage_parts(os.path.join(BASE_DIR, "val", "32.jpg"))
