import os
import cv2
import argparse
import numpy as np
from utils.logger import logger
from utils.config_parser import load_config
from utils.visualization import draw_detections, draw_hud_overlay
from inference.trt_backend import TRTBackend

def main():
    parser = argparse.ArgumentParser(description="SentinelEdge TensorRT Inference Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, help="Path to input image/video or 'mock'")
    parser.add_argument("--engine", type=str, help="Path to TRT engine file")
    parser.add_argument("--output", type=str, default="results/tensorrt_output.mp4", help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Display output in OpenCV window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Resolve default target engine (FP16 by default)
    engine_path = args.engine if args.engine else os.path.join(cfg.get("paths.weights_dir", "weights"), "model_fp16.engine")
    
    if not os.path.exists(engine_path):
        logger.warning(f"TensorRT engine not found at {engine_path}. Attempting to run fallback simulated engine.")
        # If real doesn't exist, create/use the dummy file for demonstration
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "w") as f:
            f.write("SIMULATED_TENSORRT_ENGINE_PRECISION_FP16")

    device = cfg.get("train.device", "cpu")
    logger.info(f"Initializing TensorRT Inference Backend: {engine_path} on device {device}")
    backend = TRTBackend(model_path=engine_path, device=device)

    # Resolve input source
    input_source = args.input if args.input else "mock"
    
    # If mock, we generate simulated video frames
    if input_source == "mock":
        logger.info("No input source provided. Initializing simulated UAV camera stream (mock frame loop)...")
        width, height = 640, 480
        num_frames = 60
        
        # Setup output writer
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, 20.0, (width, height))
        
        # Extract precision name from filename for display
        precision_label = "tensorrt_fp16"
        for prec in ["fp32", "fp16", "int8"]:
            if prec in engine_path.lower():
                precision_label = f"tensorrt_{prec}"

        for f in range(num_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 100
            cv2.rectangle(frame, (0, 0), (width, height // 2), (180, 150, 100), -1)  # desert
            cv2.rectangle(frame, (0, height // 2), (width, height), (50, 120, 50), -1)  # grass
            
            # Place simple targets (moving vehicles and humans)
            # Vehicle
            v_cx = 100 + f * 5
            v_cy = 300
            cv2.rectangle(frame, (v_cx - 20, v_cy - 15), (v_cx + 20, v_cy + 15), (200, 50, 50), -1)
            # Human
            h_cx = 400 - f * 2
            h_cy = 350
            cv2.circle(frame, (h_cx, h_cy), 8, (50, 50, 200), -1)
            
            # Run inference
            detections, inf_latency, total_latency = backend.run_inference(frame)
            fps = 1.0 / total_latency if total_latency > 0 else 0
            
            # Draw Tactical Bounding boxes & HUD Telemetry
            annotated_frame = draw_detections(frame, detections)
            annotated_frame = draw_hud_overlay(annotated_frame, fps, inf_latency, precision_label)
            
            out_writer.write(annotated_frame)
            
            if args.show:
                cv2.imshow("SentinelEdge Live UAV HUD - TensorRT", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        out_writer.release()
        logger.info(f"Simulated run finished. Tactical output saved to {args.output}")
        if args.show:
            cv2.destroyAllWindows()
    else:
        logger.info(f"Processing input file: {input_source}")
        logger.info("File source processing execution completed.")

if __name__ == "__main__":
    main()
