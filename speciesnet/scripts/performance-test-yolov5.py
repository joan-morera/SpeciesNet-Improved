"""Performance testing script for the SpeciesNet model.

Purpose:
This script is designed to measure the performance of the SpeciesNet model in an
isolated environment. It is intended for comparing different model versions
(e.g., YOLOv5 vs. YOLOv10) when run in separate CI/CD environments. The version of
the underlying model is determined by the 'ultralytics' library and model weights
present in the active environment, not by dynamic loading within this script.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from speciesnet import DEFAULT_MODEL
from speciesnet.multiprocessing import SpeciesNet

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def run_predictions(model: SpeciesNet, image_paths: list[str]) -> list[float]:
    """Runs predictions and returns a list of inference times."""
    inference_times = []
    for filepath in image_paths:
        try:
            instances_dict = {"instances": [{"filepath": filepath}]}
            inference_start_time = time.time()
            _ = model.predict(instances_dict=instances_dict, run_mode="single_thread")
            inference_end_time = time.time()
            inference_times.append(inference_end_time - inference_start_time)
        except Exception as e:
            print(f"Error during prediction on {filepath}: {e}", file=sys.stderr)
    return inference_times

def main():
    """
    Main function to parse arguments and run the performance test.
    """
    parser = argparse.ArgumentParser(
        description="Measure SpeciesNet model performance for YOLOv5.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
        help="Path to the directory containing images for performance testing."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="performance_results.json",
        help="Path to save the performance results in JSON format. (Default: performance_results.json)"
    )
    parser.add_argument(
        "--num_warmup_runs",
        type=int,
        default=5,
        help="Number of initial runs to warm up the model and device. (Default: 5)"
    )
    parser.add_argument(
        "--num_measurement_runs",
        type=int,
        default=20,
        help="Number of times to run inference on the entire dataset for measurement. (Default: 20)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="SpeciesNet model to load."
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_dir():
        print(f"Error: Dataset path '{args.dataset_path}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    image_extensions = [".jpg", ".jpeg", ".png"]
    image_paths = sorted([p for p in dataset_path.glob("**/*") if p.suffix.lower() in image_extensions])

    if not image_paths:
        print(f"Error: No images found in '{args.dataset_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images for testing.")

    print("Loading SpeciesNet model...")
    try:
        model = SpeciesNet(args.model)
    except Exception as e:
        print(f"Error loading SpeciesNet: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded.")

    image_paths_str = [str(p) for p in image_paths]

    print(f"\nStarting {args.num_warmup_runs} warmup runs...")
    for i in range(args.num_warmup_runs):
        run_predictions(model, image_paths_str)
        print(f"Warmup run {i + 1}/{args.num_warmup_runs} completed.")

    print(f"\nStarting {args.num_measurement_runs} measurement runs...")
    all_inference_times = []
    total_measurement_start_time = time.time()

    for i in range(args.num_measurement_runs):
        run_start_time = time.time()
        inference_times = run_predictions(model, image_paths_str)
        all_inference_times.extend(inference_times)
        run_end_time = time.time()
        print(f"Measurement run {i + 1}/{args.num_measurement_runs} completed in {run_end_time - run_start_time:.2f} seconds.")

    total_measurement_end_time = time.time()
    total_execution_time = total_measurement_end_time - total_measurement_start_time

    if not all_inference_times:
        print("Error: No successful inference times were recorded.", file=sys.stderr)
        sys.exit(1)

    total_predictions = len(all_inference_times)
    average_inference_time = sum(all_inference_times) / total_predictions
    images_per_second = len(image_paths) * args.num_measurement_runs / total_execution_time if total_execution_time > 0 else 0

    results = {
        "model_name": "MegaDetectorV5a",
        "dataset_path": str(dataset_path),
        "num_images": len(image_paths),
        "num_warmup_runs": args.num_warmup_runs,
        "num_measurement_runs": args.num_measurement_runs,
        "total_successful_predictions": total_predictions,
        "metrics": {
            "total_execution_time_sec": total_execution_time,
            "average_inference_time_per_image_sec": average_inference_time,
            "images_per_second_ips": images_per_second,
        }
    }

    output_path = Path(args.output_file)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nPerformance results saved to '{output_path}'")
    except IOError as e:
        print(f"Error writing to output file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
