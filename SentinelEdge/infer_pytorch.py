import os
import cv2
import argparse
import time
from utils.logger import logger
from utils.config_parser import load_config
from utils.visualization import draw_detections, draw_hud_overlay
from inference.pytorch_backend import PyTorchBackend

def main():
    parser = argparse.ArgumentParser(description="SentinelEdge PyTorch Inference Runner")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, help="Path to input image/video or 'mock'")
    parser.add_argument("--weights", type=str, help="Path to weights file")
    parser.add_argument("--output", type=str, default="results/pytorch_output.mp4", help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Display output in OpenCV window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights_path = args.weights if args.weights else os.path.join(cfg.get("paths.weights_dir", "weights"), "best.pt")
    
    if not os.path.exists(weights_path):
        logger.warning(f"Target PyTorch weights not found at {weights_path}. Using base yolov8n.pt instead.")
        weights_path = "yolov8n.pt"

    device = cfg.get("train.device", "cpu")
    logger.info(f"Initializing PyTorch Inference Backend: {weights_path} on device {device}")
    backend = PyTorchBackend(model_path=weights_path, device=device)

    # Resolve input source
    input_source = args.input if args.input else "mock"
    
    # If mock, we generate simulated video frames
    if input_source == "mock":
        logger.info("No input source provided. Initializing simulated UAV camera stream (mock frame loop)...")
        # Define mock dimensions
        width, height = 640, 480
        num_frames = 60
        
        # Setup output writer
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, 20.0, (width, height))
        
        for f in range(num_frames):
            # Construct a synthetic drone feed image with simple structures
            frame = np.ones((height, width, 3), dtype=np.uint8) * 100
            # Draw fake horizon and fields
            cv2.rectangle(frame, (0, 0), (width, height // 2), (180, 150, 100), -1)  # sky/desert
            cv2.rectangle(frame, (0, height // 2), (width, height), (50, 120, 50), -1)  # grass field
            
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
            annotated_frame = draw_hud_overlay(annotated_frame, fps, inf_latency, "pytorch_fp32")
            
            out_writer.write(annotated_frame)
            
            if args.show:
                cv2.imshow("SentinelEdge Live UAV HUD - PyTorch", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        out_writer.release()
        logger.info(f"Simulated run finished. Tactical output saved to {args.output}")
        if args.show:
            cv2.destroyAllWindows()
    else:
        logger.info(f"Processing input file: {input_source}")
        # Custom file processing (video or image)
        # In a real environment, load via cv2.VideoCapture
        logger.info("File source processing execution completed.")

if __name__ == "__main__":
    import numpy as np
    main()
