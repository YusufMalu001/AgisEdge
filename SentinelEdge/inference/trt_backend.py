import os
import cv2
import time
import numpy as np
from inference.base import InferenceBackend
from utils.logger import logger

# Attempt to load pycuda and tensorrt
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    pass

class TRTBackend(InferenceBackend):
    """
    Inference backend using custom TensorRT C++ wrappers/bindings in Python.
    Includes hardware fallback simulation.
    """
    
    def load_model(self):
        # Detect if model is simulated
        self.is_simulated = True
        if HAS_TRT and os.path.exists(self.model_path):
            # Check if it's the dummy simulated engine file
            with open(self.model_path, "r", errors='ignore') as f:
                header = f.read(50)
                if "SIMULATED_TENSORRT_ENGINE" not in header:
                    self.is_simulated = False

        if self.is_simulated:
            logger.warning(f"TensorRT Engine Backend running in SIMULATION fallback mode for file {self.model_path}")
            self.net_h, self.net_w = 640, 640
            return

        # Real TensorRT engine deserialization
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        # Parse inputs/outputs binding shapes and allocate buffers
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * 1
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Host & device allocations
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.allocations.append(int(device_mem))
            
            binding_info = {
                "name": binding,
                "host": host_mem,
                "device": device_mem,
                "shape": self.engine.get_binding_shape(binding),
                "dtype": dtype
            }
            
            if self.engine.binding_is_input(binding):
                self.inputs.append(binding_info)
                self.net_h = binding_info["shape"][2]
                self.net_w = binding_info["shape"][3]
            else:
                self.outputs.append(binding_info)

    def preprocess(self, img):
        if self.is_simulated:
            # Simulate slight CPU processing overhead
            time.sleep(0.001)
            return img
            
        # Standard letterbox resizing
        h, w = img.shape[:2]
        scale = min(self.net_h / h, self.net_w / w)
        nh = int(h * scale)
        nw = int(w * scale)
        
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.net_h, self.net_w, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw, :] = resized
        
        input_data = canvas[:, :, ::-1].transpose(2, 0, 1)  # HWC -> CHW
        input_data = np.ascontiguousarray(input_data).astype(np.float32) / 255.0
        self.last_padding = (scale, nw, nh)
        return input_data

    def predict(self, preprocessed_input):
        if self.is_simulated:
            # Simulate GPU TensorRT execution latency: FP16/INT8 speeds are super fast
            # FP32: ~5-15ms, FP16: ~2-5ms, INT8: ~1-3ms
            latency_sim = 0.003  # 3ms default simulation speed
            if "FP32" in self.model_path:
                latency_sim = 0.012
            elif "FP16" in self.model_path:
                latency_sim = 0.004
            elif "INT8" in self.model_path:
                latency_sim = 0.002
            time.sleep(latency_sim)
            return None

        # Real inference run
        # Copy host memory input to device GPU memory
        np.copyto(self.inputs[0]["host"], preprocessed_input.ravel())
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])
        
        # Execute TRT forward pass
        self.context.execute_v2(self.allocations)
        
        # Retrieve prediction results back to CPU
        for out in self.outputs:
            cuda.memcpy_dtoh(out["host"], out["device"])
            
        return [out["host"].reshape(out["shape"]) for out in self.outputs]

    def postprocess(self, raw_output, original_shape, conf_threshold=0.25, iou_threshold=0.45):
        if self.is_simulated:
            # Return synthetic drone detections for testing the end-to-end framework
            # Simulate 1-3 random detections
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

        # Real postprocessing logic (matches ONNX backend postprocessing)
        raw_output_tensor = raw_output[0]  # shape: [1, 7, 8400]
        output = np.squeeze(raw_output_tensor)  # shape: [7, 8400]
        output = output.T  # shape: [8400, 7]
        
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []

        x_center, y_center, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - bw / 2
        y1 = y_center - bh / 2
        x2 = x_center + bw / 2
        y2 = y_center + bh / 2
        
        scale, nw, nh = self.last_padding
        x1 = np.clip(x1, 0, nw) / scale
        y1 = np.clip(y1, 0, nh) / scale
        x2 = np.clip(x2, 0, nw) / scale
        y2 = np.clip(y2, 0, nh) / scale
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)
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
