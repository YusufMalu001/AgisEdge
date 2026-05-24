import os
import argparse
import numpy as np
from utils.logger import logger
from utils.config_parser import load_config

# Attempt to import TensorRT and PyCUDA, handle fallback gracefully
HAS_TRT = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TRT = True
except ImportError:
    logger.warning("TensorRT or PyCUDA Python packages not installed. TensorRT engine builder will run in SIMULATED fallback mode.")

class INT8EntropyCalibrator:
    """
    Simulated or functional INT8 Entropy Calibrator for TensorRT.
    In real usage, it feeds calibration images to the engine builder to determine optimal scale factors for INT8 quantization.
    """
    def __init__(self, calibration_images_dir, cache_file, batch_size=8, img_size=(640, 640)):
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.img_size = img_size
        # Collect paths to calibration images
        self.image_paths = []
        if os.path.exists(calibration_images_dir):
            self.image_paths = [
                os.path.join(calibration_images_dir, f)
                for f in os.listdir(calibration_images_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        self.num_batches = len(self.image_paths) // batch_size
        self.current_batch = 0
        
        # If functional TRT exists, we'd inherit from trt.IInt8EntropyCalibrator2
        # However, to avoid import errors on CPU machines, we dynamic-bind or mock it.
        if HAS_TRT:
            # We dynamically inject the calibrator base class if TRT is present
            class TRTCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, outer):
                    trt.IInt8EntropyCalibrator2.__init__(self)
                    self.outer = outer
                    self.device_input = cuda.mem_alloc(self.outer.batch_size * 3 * self.outer.img_size[0] * self.outer.img_size[1] * 4)

                def get_batch_size(self):
                    return self.outer.batch_size

                def get_batch(self, names):
                    if self.outer.current_batch >= self.outer.num_batches:
                        return None
                    # Load and preprocess batch images
                    batch_data = []
                    for i in range(self.outer.batch_size):
                        idx = self.outer.current_batch * self.outer.batch_size + i
                        # Dummy or real preprocessing
                        import cv2
                        img = cv2.imread(self.outer.image_paths[idx])
                        img = cv2.resize(img, self.outer.img_size)
                        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                        batch_data.append(img)
                    
                    batch_data = np.stack(batch_data).ravel()
                    cuda.memcpy_htod(self.device_input, batch_data)
                    self.outer.current_batch += 1
                    return [int(self.device_input)]

                def read_calibration_cache(self):
                    if os.path.exists(self.outer.cache_file):
                        with open(self.outer.cache_file, "rb") as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache):
                    with open(self.outer.cache_file, "wb") as f:
                        f.write(cache)

            self.calibrator_impl = TRTCalibrator(self)
        else:
            self.calibrator_impl = None

def build_engine(onnx_path, engine_path, precision="fp16", workspace_size=1073741824, calib_dir=None):
    """
    Builds a TensorRT Engine from an ONNX model.
    Supported precisions: 'fp32', 'fp16', 'int8'
    """
    logger.info(f"Initiating TensorRT engine build: ONNX={onnx_path} -> Engine={engine_path} (Precision={precision})")

    # Explanation of TensorRT optimizations:
    # 1. Quantization: Mapping FP32 weights (32-bit float) to FP16 or INT8 (8-bit integer) to reduce latency and memory bandwidth.
    # 2. Kernel Fusion: Combining multiple operations (e.g. Conv + Bias + ReLU) into a single optimized GPU execution kernel, reducing memory round-trips.
    # 3. Reduced Precision Inference: Exploiting Tensor Cores on NVIDIA architecture (Volta, Turing, Ampere, Ada Lovelace) for matrix multiply operations.
    # 4. Dynamic Tensor Memory Management: Optimizing workspace and tensor memory allocation reuse.

    if not HAS_TRT:
        logger.info("[SIMULATION MODE] Simulating TensorRT optimizations...")
        logger.info(f"[SIMULATION] Step 1: Parsing ONNX model file from {onnx_path}")
        logger.info(f"[SIMULATION] Step 2: Applying GPU Layer/Kernel Fusion on simulated layers")
        logger.info(f"[SIMULATION] Step 3: Calibrating precision parameters for configuration '{precision}'")
        if precision == "int8":
            logger.info(f"[SIMULATION] Step 3b: Creating calibration cache using images from {calib_dir}")
        logger.info(f"[SIMULATION] Step 4: Serializing engine structure using target workspace size {workspace_size} bytes")
        
        # Write a dummy engine file for pipeline testing
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "w") as f:
            f.write(f"SIMULATED_TENSORRT_ENGINE_PRECISION_{precision.upper()}")
        logger.info(f"[SIMULATION] Simulated engine file generated at: {engine_path}")
        return True

    # Real TensorRT build logic
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Parse ONNX Network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model file not found at {onnx_path}")
        return False
        
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX Parser Error: {parser.get_error(error)}")
            return False

    # Set builder configurations
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # Configure precision modes
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("Target platform does not support FP16 hardware acceleration. Falling back to FP32.")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 builder flag enabled.")
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("Target platform does not support INT8 hardware acceleration. Falling back to FP16/FP32.")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 builder flag enabled. Setting up calibrator...")
            cache_file = engine_path + ".calibration_cache"
            calibrator = INT8EntropyCalibrator(calib_dir, cache_file)
            if calibrator.calibrator_impl:
                config.int8_calibrator = calibrator.calibrator_impl
                logger.info(f"Calibration dataset configured. Cache will be written to {cache_file}")
            else:
                logger.error("Failed to construct TRT calibrator object.")

    # Build and serialize engine
    logger.info("Building CUDA engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        logger.error("Failed to build TensorRT serialized network engine!")
        return False

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    logger.info(f"Serialized TensorRT engine successfully saved to: {engine_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="SentinelEdge TensorRT Optimization Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8"], default="fp16", help="Engine precision target")
    parser.add_argument("--onnx", type=str, help="Override source ONNX path")
    parser.add_argument("--output", type=str, help="Override target engine path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    onnx_path = args.onnx if args.onnx else os.path.join(cfg.get("paths.export_dir", "export"), "model.onnx")
    
    # If source ONNX doesn't exist, log warning
    if not os.path.exists(onnx_path):
        logger.warning(f"Source ONNX model not found at {onnx_path}. Ensure export phase is completed.")
        
    engine_name = f"model_{args.precision}.engine"
    output_path = args.output if args.output else os.path.join(cfg.get("paths.weights_dir", "weights"), engine_name)
    
    workspace_size = cfg.get("tensorrt.workspace_size", 1073741824)
    calib_dir = cfg.get("tensorrt.calibration_dataset", "datasets/calibration")

    build_engine(
        onnx_path=onnx_path,
        engine_path=output_path,
        precision=args.precision,
        workspace_size=workspace_size,
        calib_dir=calib_dir
    )

if __name__ == "__main__":
    main()
