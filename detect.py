# detect.py
import os
import cv2
import numpy as np
import torch
from PIL import Image
import streamlit as st
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
import tempfile

# Constants
DAMAGE_CLASSES = {0: 'damage'}
PART_CLASSES = {
    0: 'headlamp',
    1: 'rear_bumper', 
    2: 'door', 
    3: 'hood', 
    4: 'front_bumper',
    5: 'windshield'
}

class CarDamageTrainer:
    def __init__(self, data_dir="archive"):
        self.data_dir = data_dir
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def register_datasets(self):
        """Register training and validation datasets"""
        try:
            # Register damage datasets
            register_coco_instances(
                "car_damage_train", {},
                f"{self.data_dir}/train/COCO_train_annos.json",
                f"{self.data_dir}/img"
            )
            register_coco_instances(
                "car_damage_val", {},
                f"{self.data_dir}/val/COCO_val_annos.json",
                f"{self.data_dir}/img"
            )
            
            # Register parts datasets
            register_coco_instances(
                "car_parts_train", {},
                f"{self.data_dir}/train/COCO_mul_train_annos.json",
                f"{self.data_dir}/img"
            )
            register_coco_instances(
                "car_parts_val", {},
                f"{self.data_dir}/val/COCO_mul_val_annos.json",
                f"{self.data_dir}/img"
            )
            return True
        except Exception as e:
            st.error(f"Error registering datasets: {str(e)}")
            return False

    def train_damage_model(self):
        """Train the damage detection model"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Dataset configuration
        cfg.DATASETS.TRAIN = ("car_damage_train",)
        cfg.DATASETS.TEST = ("car_damage_val",)
        cfg.DATALOADER.NUM_WORKERS = 2
        
        # Model configuration
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only damage class
        cfg.OUTPUT_DIR = os.path.join(self.output_dir, "damage")
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # Save final model
        torch.save(trainer.model.state_dict(), "models/damage_model.pth")
        return os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    def train_parts_model(self):
        """Train the parts segmentation model"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Dataset configuration
        cfg.DATASETS.TRAIN = ("car_parts_train",)
        cfg.DATASETS.TEST = ("car_parts_val",)
        cfg.DATALOADER.NUM_WORKERS = 2
        
        # Model configuration
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1500  # More iterations for more classes
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # 5 parts + background
        cfg.OUTPUT_DIR = os.path.join(self.output_dir, "parts")
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        # Save final model
        torch.save(trainer.model.state_dict(), "models/parts_model.pth")
        return os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

class CarDamageDetector:
    def __init__(self):
        self.damage_predictor = None
        self.part_predictor = None
        self.models_loaded = False
        
    def load_models(self):
        """Load trained models"""
        try:
            os.makedirs("models", exist_ok=True)
            
            # Damage model
            damage_cfg = get_cfg()
            damage_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            damage_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
            damage_cfg.MODEL.WEIGHTS = "models/damage_model.pth"
            damage_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            damage_cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.damage_predictor = DefaultPredictor(damage_cfg)
            
            # Parts model
            parts_cfg = get_cfg()
            parts_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            parts_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
            parts_cfg.MODEL.WEIGHTS = "models/parts_model.pth"
            parts_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            parts_cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.part_predictor = DefaultPredictor(parts_cfg)
            
            self.models_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def analyze_image(self, image):
        """Analyze an image for damage"""
        if not self.models_loaded:
            return None
            
        try:
            image_np = np.array(image)
            
            # Get predictions
            damage_output = self.damage_predictor(image_np)
            parts_output = self.part_predictor(image_np)
            
            # Visualize results
            v = Visualizer(image_np[:, :, ::-1], scale=0.8)
            out = v.draw_instance_predictions(damage_output["instances"].to("cpu"))
            out = v.draw_instance_predictions(parts_output["instances"].to("cpu"))
            
            # Match damages to parts
            damaged_parts = self.match_damage_to_parts(
                damage_output["instances"],
                parts_output["instances"]
            )
            
            return {
                "visualization": out.get_image()[:, :, ::-1],
                "damaged_parts": damaged_parts
            }
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None
    
    def match_damage_to_parts(self, damage_instances, parts_instances):
        """Match damage areas to car parts"""
        damage_boxes = damage_instances.pred_boxes.tensor.cpu().numpy()
        parts_boxes = parts_instances.pred_boxes.tensor.cpu().numpy()
        parts_classes = parts_instances.pred_classes.cpu().numpy()
        
        damaged_parts = set()
        for d_box in damage_boxes:
            d_center = [(d_box[0] + d_box[2])/2, (d_box[1] + d_box[3])/2]
            min_dist = float('inf')
            closest_part = None
            
            for p_box, p_class in zip(parts_boxes, parts_classes):
                p_center = [(p_box[0] + p_box[2])/2, (p_box[1] + p_box[3])/2]
                dist = np.linalg.norm(np.array(d_center) - np.array(p_center))
                if dist < min_dist:
                    min_dist = dist
                    closest_part = PART_CLASSES.get(int(p_class), "unknown")
            
            if closest_part:
                damaged_parts.add(closest_part)
        
        return list(damaged_parts)

def main():
    st.title("🚗 Car Damage Detection System")
    
    # Initialize trainer and detector
    trainer = CarDamageTrainer()
    detector = CarDamageDetector()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Train Models", "Detect Damage"])
    
    with tab1:
        st.header("Train Models")
        if st.button("Register Datasets"):
            if trainer.register_datasets():
                st.success("Datasets registered successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Damage Model")
            if st.button("Train Damage Model"):
                with st.spinner("Training damage model..."):
                    model_path = trainer.train_damage_model()
                    st.success(f"Damage model trained and saved to: {model_path}")
        
        with col2:
            st.subheader("Parts Model")
            if st.button("Train Parts Model"):
                with st.spinner("Training parts model..."):
                    model_path = trainer.train_parts_model()
                    st.success(f"Parts model trained and saved to: {model_path}")
    
    with tab2:
        st.header("Detect Damage")
        if st.button("Load Models"):
            with st.spinner("Loading models..."):
                if detector.load_models():
                    st.session_state.models_loaded = True
                    st.success("Models loaded successfully!")
                else:
                    st.error("Failed to load models")
        
        uploaded_files = st.file_uploader(
            "Upload car images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.session_state.get("models_loaded", False):
            for uploaded_file in uploaded_files:
                col1, col2 = st.columns(2)
                
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Original", use_container_width=True)
                
                with col2:
                    result = detector.analyze_image(image)
                    if result:
                        st.image(
                            result["visualization"],
                            caption="Analysis",
                            use_container_width=True
                        )
                        if result["damaged_parts"]:
                            st.write("**Damaged Parts:**")
                            for part in result["damaged_parts"]:
                                st.write(f"- {part}")
                        else:
                            st.success("No damage detected")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    main()