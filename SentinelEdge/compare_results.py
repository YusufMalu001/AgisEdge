import os
import argparse
import pandas as pd
from utils.logger import logger
from utils.config_parser import load_config

def main():
    parser = argparse.ArgumentParser(description="SentinelEdge Benchmark Comparison Utility")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg.get("paths.results_dir", "results")
    csv_path = os.path.join(results_dir, "benchmark_report.csv")

    if not os.path.exists(csv_path):
        logger.error(f"Benchmark report file not found at {csv_path}. Please execute 'benchmark.py' first.")
        return

    # Load results
    df = pd.read_csv(csv_path)

    # Format values for printing
    df_print = df.copy()
    df_print["Load Time (s)"] = df_print["Load Time (s)"].map(lambda x: f"{x:.4f}s")
    df_print["Avg Latency (ms)"] = df_print["Avg Latency (ms)"].map(lambda x: f"{x:.2f} ms")
    df_print["Std Latency (ms)"] = df_print["Std Latency (ms)"].map(lambda x: f"{x:.2f} ms")
    df_print["P95 Latency (ms)"] = df_print["P95 Latency (ms)"].map(lambda x: f"{x:.2f} ms")
    df_print["Throughput (FPS)"] = df_print["Throughput (FPS)"].map(lambda x: f"{x:.1f} FPS")
    df_print["Avg CPU (%)"] = df_print["Avg CPU (%)"].map(lambda x: f"{x:.1f}%")
    df_print["Avg RAM (MB)"] = df_print["Avg RAM (MB)"].map(lambda x: f"{x:.1f} MB")
    df_print["Avg GPU Mem (MB)"] = df_print["Avg GPU Mem (MB)"].map(lambda x: f"{x:.1f} MB")

    # Construct Markdown table
    headers = df_print.columns.tolist()
    header_str = "| " + " | ".join(headers) + " |"
    separator_str = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    rows = []
    for _, row in df_print.iterrows():
        rows.append("| " + " | ".join(row.astype(str).tolist()) + " |")
        
    markdown_table = "\n".join([header_str, separator_str] + rows)

    # Print Table
    print("\n" + "="*80)
    print("                    SENTINELEDGE PIPELINE OPTIMIZATION REPORT                    ")
    print("="*80)
    print(markdown_table)
    print("="*80 + "\n")

    # Performance analysis calculations
    pytorch_row = df[df["Backend"] == "PyTorch FP32"]
    trt_fp16_row = df[df["Backend"] == "TensorRT FP16"]
    trt_int8_row = df[df["Backend"] == "TensorRT INT8"]

    if not pytorch_row.empty and not trt_fp16_row.empty:
        py_lat = pytorch_row.iloc[0]["Avg Latency (ms)"]
        fp16_lat = trt_fp16_row.iloc[0]["Avg Latency (ms)"]
        
        if fp16_lat > 0:
            speedup = py_lat / fp16_lat
            print(f"[ANALYSIS] TensorRT FP16 achieved a {speedup:.2f}x speedup compared to standard PyTorch FP32.")
            
    if not pytorch_row.empty and not trt_int8_row.empty:
        py_lat = pytorch_row.iloc[0]["Avg Latency (ms)"]
        int8_lat = trt_int8_row.iloc[0]["Avg Latency (ms)"]
        
        if int8_lat > 0:
            speedup_int8 = py_lat / int8_lat
            print(f"[ANALYSIS] TensorRT INT8 achieved a {speedup_int8:.2f}x speedup compared to standard PyTorch FP32.")

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
