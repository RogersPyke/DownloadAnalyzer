"""
Purpose: Simple entry point for external scripts to trigger download statistics.
Usage: python run_stat.py [--org ORG_NAME] [--ms_org MS_ORG_NAME] [--output OUTPUT_DIR]
"""

import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import HuggingFaceDatasetAnalyzer, ModelScopeDatasetAnalyzer


def run_stat(
    org_name: str = "RoboCOIN",
    ms_org_name: str = "RoboCOIN",
    output_dir: str = "/down_stat",
    max_workers: int = 8,
) -> dict:
    """
    @input: org_name [str], ms_org_name [str], output_dir [str], max_workers [int]
    @output: dict with paths to generated files
    @scenario: External callable for download statistics generation
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(output_dir, timestamp)
    os.makedirs(save_root, exist_ok=True)

    hf_analyzer = HuggingFaceDatasetAnalyzer(org_name, save_root, max_workers)
    hf_df = hf_analyzer.run()

    ms_analyzer = ModelScopeDatasetAnalyzer(ms_org_name, save_root)
    ms_df = ms_analyzer.run()
    ms_analyzer.visualize_top_datasets()

    hf_total = (
        int(hf_df["downloads"].sum()) if hf_df is not None and not hf_df.empty else 0
    )
    ms_total = (
        int(ms_df["downloads"].sum()) if ms_df is not None and not ms_df.empty else 0
    )

    return {
        "save_dir": save_root,
        "hf_csv": os.path.join(save_root, "huggingface_dataset_downloads.csv"),
        "ms_csv": os.path.join(save_root, "modelscope_dataset_downloads.csv"),
        "hf_plot": os.path.join(save_root, "huggingface_top_datasets_plot.png"),
        "ms_plot": os.path.join(save_root, "modelscope_top_datasets_plot.png"),
        "hf_total_downloads": hf_total,
        "ms_total_downloads": ms_total,
        "combined_total": hf_total + ms_total,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run download statistics")
    parser.add_argument(
        "--org", type=str, default="RoboCOIN", help="HuggingFace org name"
    )
    parser.add_argument(
        "--ms_org", type=str, default="RoboCOIN", help="ModelScope org name"
    )
    parser.add_argument(
        "--output", type=str, default="/down_stat", help="Output directory"
    )
    parser.add_argument("--max_workers", type=int, default=8, help="Max worker threads")

    args = parser.parse_args()

    result = run_stat(
        org_name=args.org,
        ms_org_name=args.ms_org,
        output_dir=args.output,
        max_workers=args.max_workers,
    )

    print(f"Output directory: {result['save_dir']}")
    print(f"HuggingFace total: {result['hf_total_downloads']:,}")
    print(f"ModelScope total: {result['ms_total_downloads']:,}")
    print(f"Combined total: {result['combined_total']:,}")
