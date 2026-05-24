import os
import argparse
import onnx
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
from utils.logger import logger
from utils.config_parser import load_config

def export_model():
    parser = argparse.ArgumentParser(description="SentinelEdge YOLOv8 ONNX Export Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, help="Override path to PyTorch model weights")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    weights_dir = cfg.get("paths.weights_dir", "weights")
    export_dir = cfg.get("paths.export_dir", "export")
    os.makedirs(export_dir, exist_ok=True)

    # Resolve model weights path
    weights_path = args.weights if args.weights else os.path.join(weights_dir, "best.pt")
    if not os.path.exists(weights_path):
        logger.warning(f"Trained weights not found at {weights_path}. Using fallback pretrained yolov8n.pt.")
        weights_path = "yolov8n.pt"

    logger.info(f"Loading weights from: {weights_path}")
    model = YOLO(weights_path)

    # Determine export params
    opset = cfg.get("export.opset", 12)
    dynamic = cfg.get("export.dynamic", True)
    half = cfg.get("export.half", False)

    logger.info(f"Exporting model to ONNX (opset={opset}, dynamic={dynamic}, half={half})...")
    
    # Why ONNX?
    # ONNX (Open Neural Network Exchange) acts as an open format built to represent machine learning models.
    # It provides model interoperability, enabling developers to easily move models between PyTorch, ONNXRuntime, TensorRT, and OpenVINO.
    # It is critical for edge deployments because it decouples the model definition from Python/PyTorch runtime, allowing high-performance C++ inference execution.
    
    exported_file_path = model.export(
        format="onnx",
        opset=opset,
        dynamic=dynamic,
        half=half
    )

    # YOLOv8 default exporter saves it next to the weights directory or project structure.
    # Move it to the designated export directory for cleaner modular project structure
    target_onnx_path = os.path.join(export_dir, "model.onnx")
    if os.path.exists(exported_file_path):
        import shutil
        shutil.move(exported_file_path, target_onnx_path)
        logger.info(f"Model successfully exported and moved to: {target_onnx_path}")
    else:
        logger.error("ONNX export succeeded but file could not be found.")
        return

    # ONNX Structure Verification
    logger.info("Verifying ONNX model structure and constraints...")
    onnx_model = onnx.load(target_onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX validation passed: File structure is correct and contains no schema violations.")

    # Display inputs/outputs structure
    for input_tensor in onnx_model.graph.input:
        logger.info(f"Model Input name: '{input_tensor.name}', Shape: {input_tensor.type.tensor_type.shape}")
    for output_tensor in onnx_model.graph.output:
        logger.info(f"Model Output name: '{output_tensor.name}', Shape: {output_tensor.type.tensor_type.shape}")

    # ONNXRuntime Testing / Smoke test
    logger.info("Initializing ONNXRuntime Inference Session check...")
    try:
        session = ort.InferenceSession(target_onnx_path, providers=["CPUExecutionProvider"])
        
        # Prepare dummy input based on model definition (YOLOv8 defaults: batch, 3, 640, 640)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Parse shape values, resolving dynamic dimensions to a static testing size
        test_shape = [1, 3, 640, 640]
        dummy_input = np.random.randn(*test_shape).astype(np.float32)
        
        logger.info(f"Executing sample inference using ONNXRuntime with dummy shape: {test_shape}...")
        outputs = session.run(None, {input_name: dummy_input})
        
        logger.info(f"ONNXRuntime test inference completed successfully. Outputs count: {len(outputs)}")
        logger.info(f"Primary Output Shape: {outputs[0].shape}")
    except Exception as e:
        logger.error(f"Failed to perform ONNXRuntime validation run: {e}")

if __name__ == "__main__":
    export_model()
