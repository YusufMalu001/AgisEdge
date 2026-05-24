import os
import argparse
from ultralytics import YOLO
from utils.logger import logger
from utils.config_parser import load_config
from utils.dataset_generator import generate_synthetic_aerial_dataset

def main():
    parser = argparse.ArgumentParser(description="SentinelEdge YOLOv8 Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--device", type=str, help="Override torch device (cpu, cuda:0)")
    args = parser.parse_args()

    # Load configurations
    cfg = load_config(args.config)
    
    # Setup directories
    dataset_dir = cfg.get("paths.dataset_dir", "datasets")
    weights_dir = cfg.get("paths.weights_dir", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Ensure dataset is present, generate synthetic if not
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    if not os.path.exists(yaml_path):
        logger.info("Dataset configuration not found. Synthesizing test UAV dataset...")
        generate_synthetic_aerial_dataset(dataset_dir=dataset_dir)

    # Determine model and training parameters
    model_type = cfg.get("train.model_type", "yolov8n")
    epochs = args.epochs if args.epochs is not None else cfg.get("train.epochs", 10)
    img_size = cfg.get("train.img_size", 640)
    batch_size = cfg.get("train.batch_size", 16)
    device = args.device if args.device is not None else cfg.get("train.device", "cpu")
    workers = cfg.get("train.workers", 4)
    optimizer = cfg.get("train.optimizer", "SGD")
    lr0 = cfg.get("train.lr0", 0.01)

    logger.info(f"Initializing YOLO model: {model_type}")
    model = YOLO(f"{model_type}.pt")

    logger.info("Starting training loop...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        project=weights_dir,
        name="train_run",
        exist_ok=True
    )

    logger.info("Training complete.")

    # Locate and copy best.pt to weights directory root
    best_weights_source = os.path.join(weights_dir, "train_run", "weights", "best.pt")
    best_weights_target = os.path.join(weights_dir, "best.pt")
    
    if os.path.exists(best_weights_source):
        import shutil
        shutil.copy(best_weights_source, best_weights_target)
        logger.info(f"Best weights copied to: {best_weights_target}")
    else:
        logger.error("Could not find trained weights checkpoint file!")

if __name__ == "__main__":
    main()
