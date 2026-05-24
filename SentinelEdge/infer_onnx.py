import os
import cv2
import argparse
import numpy as np
from utils.logger import logger
from utils.config_parser import load_config
from utils.visualization import draw_detections, draw_hud_overlay
from inference.onnx_backend import ONNXBackend

def main():
    parser = argparse.ArgumentParser(description="SentinelEdge ONNXRuntime Inference Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, help="Path to input image/video or 'mock'")
    parser.add_argument("--model", type=str, help="Path to ONNX model file")
    parser.add_argument("--output", type=str, default="results/onnx_output.mp4", help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Display output in OpenCV window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model if args.model else os.path.join(cfg.get("paths.export_dir", "export"), "model.onnx")
    
    if not os.path.exists(model_path):
        logger.error(f"ONNX model file not found at {model_path}. Please execute export phase first.")
        return

    device = cfg.get("train.device", "cpu")
    logger.info(f"Initializing ONNXRuntime Inference Backend: {model_path} on device {device}")
    backend = ONNXBackend(model_path=model_path, device=device)

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
            annotated_frame = draw_hud_overlay(annotated_frame, fps, inf_latency, "onnxruntime")
            
            out_writer.write(annotated_frame)
            
            if args.show:
                cv2.imshow("SentinelEdge Live UAV HUD - ONNXRuntime", annotated_frame)
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
