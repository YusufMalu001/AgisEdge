import os
import time
import argparse
import psutil
import pandas as pd
from utils.logger import logger
from utils.config_parser import load_config

class EdgeSimulator:
    """
    Simulates resource constraints, power profiles, and compute limitations of edge SoCs (e.g. Jetson Nano/TX2/Orin).
    Allows modeling latency vs throughput tradeoffs before committing to hardware deployment.
    """
    
    def __init__(self, config_path="configs/default.yaml"):
        self.cfg = load_config(config_path)
        self.power_mode = self.cfg.get("edge_simulation.active_mode", "balanced")
        self.cpu_limit = self.cfg.get("edge_simulation.cpu_limit_percent", 50)
        self.mem_limit = self.cfg.get("edge_simulation.memory_limit_mb", 2048)

    def set_power_mode(self, mode):
        """
        Adjusts simulated compute capabilities based on SoC power envelopes:
        - max_performance (e.g. Orin 15W/30W mode) -> No throttle
        - balanced (e.g. Orin 15W mode) -> Minor throttle
        - low_power (e.g. Jetson Nano 5W mode) -> High throttle
        """
        self.power_mode = mode
        logger.info(f"SoC Power Mode set to: {mode.upper()}")

    def run_simulated_inference(self, base_latency, img_size=(640, 640)):
        """
        Applies modifiers to the base latency based on simulated constraints.
        """
        latency_mod = base_latency
        
        # 1. Power Profile Throttle Modifier
        if self.power_mode == "low_power":
            # Jetson Nano downclocked core CPU/GPU latency penalty (e.g., 2.5x slower)
            latency_mod *= 2.5
        elif self.power_mode == "balanced":
            # Standard balanced profile (e.g., 1.2x slower)
            latency_mod *= 1.2
            
        # 2. Resolution Scaling Modifier
        # Processing 1280x1280 (4x pixels) vs 640x640 vs 320x320 (0.25x pixels)
        pixel_count = img_size[0] * img_size[1]
        base_pixels = 640 * 640
        res_factor = pixel_count / base_pixels
        
        # Latency scales approximately linearly with pixel count for preprocessing and operations
        latency_mod *= (0.7 + 0.3 * res_factor)

        # 3. Simulate process load / throttling CPU sleep
        time.sleep(latency_mod)
        
        # Monitor actual memory utilization and warn if it breaches simulated constraints
        process = psutil.Process(os.getpid())
        ram_used = process.memory_info().rss / (1024 * 1024)
        if ram_used > self.mem_limit:
            # Simulated page swap overhead penalty
            latency_mod += 0.05
            logger.warning(f"[OVERALLOCATION] Memory usage ({ram_used:.1f} MB) exceeds Edge budget ({self.mem_limit} MB). Simulating swapping latency.")

        return latency_mod

def run_tradeoff_analysis(config_path="configs/default.yaml"):
    logger.info("Initializing UAV Edge AI Deployment Tradeoff Analysis...")
    simulator = EdgeSimulator(config_path)
    
    # Configuration profiles to test
    resolutions = [320, 640, 1280]
    power_modes = ["low_power", "balanced", "max_performance"]
    base_latencies = {"FP32": 0.015, "FP16": 0.005, "INT8": 0.002}  # base 640x640 latencies in seconds

    results = []

    for mode in power_modes:
        simulator.set_power_mode(mode)
        for precision, base_lat in base_latencies.items():
            for res in resolutions:
                # Simulating 50 frames per config setup
                latencies = []
                for _ in range(50):
                    lat = simulator.run_simulated_inference(base_lat, img_size=(res, res))
                    latencies.append(lat)
                    
                avg_lat_ms = (sum(latencies) / len(latencies)) * 1000
                fps = 1000 / avg_lat_ms if avg_lat_ms > 0 else 0
                
                # Approximate UAV Detection accuracy tradeoff calculation:
                # Higher resolutions retain smaller target classes better
                # INT8 might lose minor decimal precision
                base_accuracy = 78.0  # MAP@0.5-0.95 baseline
                res_acc_mod = -15.0 if res == 320 else (0.0 if res == 640 else 6.0)
                prec_acc_mod = 0.0 if precision == "FP32" else (-0.2 if precision == "FP16" else -1.5)
                simulated_map = max(10.0, min(99.0, base_accuracy + res_acc_mod + prec_acc_mod))

                results.append({
                    "Power Mode": mode,
                    "Precision": precision,
                    "Resolution": f"{res}x{res}",
                    "Simulated Latency (ms)": avg_lat_ms,
                    "Simulated FPS": fps,
                    "Simulated Accuracy (mAP %)": simulated_map
                })

    # Save tradeoff reports
    df = pd.DataFrame(results)
    output_dir = load_config(config_path).get("paths.results_dir", "results")
    csv_path = os.path.join(output_dir, "edge_tradeoff_analysis.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Edge deployment tradeoff analysis exported to: {csv_path}")

if __name__ == "__main__":
    run_tradeoff_analysis()
