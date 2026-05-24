import os
import random
import cv2
import numpy as np
from utils.logger import logger

def generate_synthetic_aerial_dataset(dataset_dir="datasets", num_train=50, num_val=15, img_size=640):
    """
    Generates a synthetic aerial surveillance dataset in YOLO format.
    Classes: 0: vehicle, 1: human, 2: drone
    """
    classes = ["vehicle", "human", "drone"]
    
    # Define directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(dataset_dir)}
train: train/images
val: val/images

names:
  0: vehicle
  1: human
  2: drone
"""
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info(f"Created dataset config at {yaml_path}")

    # Helper function to generate images and labels
    def create_samples(split, count):
        logger.info(f"Generating {count} synthetic {split} samples...")
        for i in range(count):
            img_name = f"aerial_{split}_{i:04d}.jpg"
            lbl_name = f"aerial_{split}_{i:04d}.txt"
            
            img_path = os.path.join(dataset_dir, split, "images", img_name)
            lbl_path = os.path.join(dataset_dir, split, "labels", lbl_name)

            # Generate random background (simulating fields/roads from drone view)
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128  # gray background
            # Add some green field patches
            for _ in range(3):
                pt1 = (random.randint(0, img_size), random.randint(0, img_size))
                pt2 = (random.randint(0, img_size), random.randint(0, img_size))
                cv2.rectangle(img, pt1, pt2, (random.randint(40, 80), random.randint(100, 150), random.randint(40, 80)), -1)

            annotations = []
            num_objects = random.randint(2, 6)
            for _ in range(num_objects):
                class_id = random.randint(0, 2)
                # drone view objects are small
                w_pixels = random.randint(20, 60)
                h_pixels = random.randint(20, 60)
                cx = random.randint(w_pixels, img_size - w_pixels)
                cy = random.randint(h_pixels, img_size - h_pixels)

                # Draw synthetic object
                # vehicle: blue rectangles, human: red circles, drone: white crosses
                if class_id == 0:  # vehicle
                    cv2.rectangle(img, (cx - w_pixels//2, cy - h_pixels//2), (cx + w_pixels//2, cy + h_pixels//2), (200, 50, 50), -1)
                elif class_id == 1:  # human
                    cv2.circle(img, (cx, cy), w_pixels//4, (50, 50, 200), -1)
                else:  # drone
                    cv2.line(img, (cx - w_pixels//2, cy), (cx + w_pixels//2, cy), (220, 220, 220), 4)
                    cv2.line(img, (cx, cy - h_pixels//2), (cx, cy + h_pixels//2), (220, 220, 220), 4)

                # Normalized YOLO format coordinates
                x_center = cx / img_size
                y_center = cy / img_size
                width = w_pixels / img_size
                height = h_pixels / img_size

                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Save image and label
            cv2.imwrite(img_path, img)
            with open(lbl_path, "w") as f:
                f.write("\n".join(annotations))

    create_samples("train", num_train)
    create_samples("val", num_val)
    logger.info("Synthetic dataset creation complete.")

if __name__ == "__main__":
    generate_synthetic_aerial_dataset()
