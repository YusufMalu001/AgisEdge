import cv2
import numpy as np
from inference.base import InferenceBackend
from ultralytics import YOLO

class PyTorchBackend(InferenceBackend):
    """
    Inference backend using raw PyTorch (Ultralytics YOLOv8 library).
    """
    
    def load_model(self):
        self.model = YOLO(self.model_path)
        # Move model to device
        self.model.to(self.device)

    def preprocess(self, img):
        # Ultralytics handles its own preprocessing, return image as-is
        return img

    def predict(self, preprocessed_input):
        # We pass verbose=False to keep logs clean
        results = self.model.predict(preprocessed_input, device=self.device, verbose=False)
        return results

    def postprocess(self, raw_output, original_shape, conf_threshold=0.25, iou_threshold=0.45):
        detections = []
        if not raw_output:
            return detections
            
        result = raw_output[0]
        boxes = result.boxes
        
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            
            detections.append([x1, y1, x2, y2, confidence, class_id])
            
        return detections
