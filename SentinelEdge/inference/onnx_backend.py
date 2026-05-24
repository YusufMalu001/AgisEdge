import cv2
import time
import numpy as np

# Handle missing onnxruntime gracefully
HAS_ORT = False
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    pass

from inference.base import InferenceBackend

class ONNXBackend(InferenceBackend):
    """
    Inference backend using ONNXRuntime with NumPy-based preprocessing & NMS.
    """
    
    def load_model(self):
        self.is_simulated = not HAS_ORT
        
        if self.is_simulated:
            from utils.logger import logger
            logger.warning(f"ONNXRuntime not installed. Running ONNXBackend in SIMULATION fallback mode for file {self.model_path}")
            self.net_h, self.net_w = 640, 640
            return

        # Configure execution providers
        providers = ["CPUExecutionProvider"]
        if self.device != "cpu":
            providers = ["CUDAExecutionProvider"] + providers
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [batch, 3, height, width]
        self.net_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
        self.net_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640

    def preprocess(self, img):
        if self.is_simulated:
            time.sleep(0.001)
            return img

        """
        Resize image to network size with letterbox (padding), normalize to [0,1],
        and transpose to [1, 3, H, W].
        """
        h, w = img.shape[:2]
        
        # Calculate letterbox scale
        scale = min(self.net_h / h, self.net_w / w)
        nh = int(h * scale)
        nw = int(w * scale)
        
        # Resize
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Create padded canvas
        canvas = np.full((self.net_h, self.net_w, 3), 114, dtype=np.uint8)
        
        # Paste resized image at (0, 0)
        canvas[:nh, :nw, :] = resized
        
        # Convert BGR to RGB, normalize, and transpose
        input_data = canvas[:, :, ::-1].transpose(2, 0, 1)  # HWC -> CHW
        input_data = np.ascontiguousarray(input_data).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dim
        
        # Save padding scale and size for postprocessing reconstruction
        self.last_padding = (scale, nw, nh)
        return input_data

    def predict(self, preprocessed_input):
        if self.is_simulated:
            time.sleep(0.010)  # Simulated ONNXRuntime latency: 10ms
            return None

        outputs = self.session.run(None, {self.input_name: preprocessed_input})
        return outputs[0]

    def postprocess(self, raw_output, original_shape, conf_threshold=0.25, iou_threshold=0.45):
        if self.is_simulated:
            # Return synthetic drone detections for testing the end-to-end framework
            num_dets = np.random.randint(1, 4)
            detections = []
            h, w = original_shape[:2]
            for _ in range(num_dets):
                dw = np.random.randint(40, 80)
                dh = np.random.randint(40, 80)
                cx = np.random.randint(dw, w - dw)
                cy = np.random.randint(dh, h - dh)
                conf = float(np.random.uniform(0.4, 0.95))
                cls_id = int(np.random.randint(0, 3))
                detections.append([cx - dw//2, cy - dh//2, cx + dw//2, cy + dh//2, conf, cls_id])
            return detections
        # raw_output shape: [1, 4 + num_classes, 8400]
        output = np.squeeze(raw_output)  # shape: [7, 8400]
        output = output.T  # shape: [8400, 7]
        
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        # Find maximum class scores and corresponding labels
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter detections by threshold
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []

        # Convert xywh to xyxy (x_center, y_center, w, h -> x1, y1, x2, y2)
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        # Reconstruct original image coordinate space using padding metadata
        scale, nw, nh = self.last_padding
        
        # Map back coordinates (clamping to the actual resized canvas box)
        x1 = np.clip(x1, 0, nw) / scale
        y1 = np.clip(y1, 0, nh) / scale
        x2 = np.clip(x2, 0, nw) / scale
        y2 = np.clip(y2, 0, nh) / scale
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)
        
        # Perform Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            conf_threshold,
            iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append([
                    float(boxes_xyxy[idx][0]),
                    float(boxes_xyxy[idx][1]),
                    float(boxes_xyxy[idx][2]),
                    float(boxes_xyxy[idx][3]),
                    float(confidences[idx]),
                    int(class_ids[idx])
                ])
                
        return detections
