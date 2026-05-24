from abc import ABC, abstractmethod
import time
import numpy as np

class InferenceBackend(ABC):
    """
    Abstract Base Class for SentinelEdge Inference Backends.
    Enforces a consistent, modular interface across all runtime engines.
    """
    
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.load_time = 0
        
        start_time = time.time()
        self.load_model()
        self.load_time = time.time() - start_time

    @abstractmethod
    def load_model(self):
        """
        Loads the model files into memory/device.
        """
        pass

    @abstractmethod
    def preprocess(self, img):
        """
        Applies resizing, normalization, color-space conversion, and batching.
        Returns the preprocessed image tensor.
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_input):
        """
        Executes raw model forward pass.
        """
        pass

    @abstractmethod
    def postprocess(self, raw_output, original_shape, conf_threshold=0.25, iou_threshold=0.45):
        """
        Applies Non-Maximum Suppression (NMS) and maps boxes back to original shape coordinates.
        Returns a list of detections: [x1, y1, x2, y2, confidence, class_id]
        """
        pass

    def run_inference(self, img, conf_threshold=0.25, iou_threshold=0.45):
        """
        Full inference pipeline including execution timing.
        Returns:
            detections (list): postprocessed detections
            latency (float): latency in seconds
        """
        t0 = time.time()
        
        # Preprocessing
        input_data = self.preprocess(img)
        
        # Predict
        t1 = time.time()
        raw_output = self.predict(input_data)
        inference_latency = time.time() - t1
        
        # Postprocessing
        detections = self.postprocess(raw_output, img.shape, conf_threshold, iou_threshold)
        
        total_latency = time.time() - t0
        return detections, inference_latency, total_latency
