#!/usr/bin/env python3
"""
Graphyne Full Pipeline

This script coordinates the complete graphyne workflow:

1. Run graphyne structure analysis (Stage 1)
2. Generate graphyne triangle clusters (Stage 2)
3. Process and deduplicate RDKit structures (Stage 3)
4. Add hydrogens to final unique clusters (Stage 4)

Usage:
    python3 run_graphyne_pipeline.py -n structure.xyz -c 8

"""
import argparse
import os
import sys

# --------------------- PATH ---------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

# --------------------- IMPORTS --------------------- #
from graphyne_analyzer import main as run_graphyne_analysis
from triangle_cluster_generator import generate_all_triangle_clusters
from graphyne_cluster_processor import main as run_rdkit_processing
from hydrogen_adder import main as run_hydrogen_addition

# --------------------- CLI --------------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Full Graphyne Pipeline")
    parser.add_argument(
        "-n", "--name",
        required=True,
        dest="input_file",
        help="Input XYZ structure file"
    )
    parser.add_argument(
        "-c", "--cpu",
        dest="n_cpu",
        type=int,
        default=1,
        help="Number of CPU cores"
    )
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        default="output",
        help="Base output directory (default: output)"
    )
    return parser.parse_args()

# --------------------- MAIN --------------------- #
def main():
    args = parse_arguments()
    input_file = args.input_file
    n_cpu = max(1, args.n_cpu)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)


    print("=" * 60)
    print("GRAPHYNE CLUSTERS GENERATION PIPELINE")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # ---------------- STAGE 1 ----------------
    print("\n[1/4] Structure analysis\n")
    connectivity_file = run_graphyne_analysis(
        input_file=input_file,
        n_cpu=n_cpu,
        output_dir=output_dir
    )

    # ---------------- STAGE 2 ----------------
    print("\n[2/4] Triangle cluster generation\n")
    generate_all_triangle_clusters(
        input_filename=connectivity_file,
        min_cluster_size=2,
        n_cpu=n_cpu,
        output_dir=output_dir
    )

    # ---------------- STAGE 3 ----------------
    print("\n[3/4] RDKit processing + deduplication\n")
    run_rdkit_processing(
        input_dir=os.path.join(output_dir, "triangle_clusters_all"),
        output_dir=output_dir,
        n_cpu=n_cpu
    )

    # ---------------- STAGE 4 ----------------
    print("\n[4/4] Hydrogen addition\n")
    run_hydrogen_addition(
        input_dir=os.path.join(output_dir, "unique_clusters"),
        output_dir=output_dir,
        n_cpu=n_cpu
    )

    print("\nPipeline completed successfully!")
    ###print(f"All outputs written to: {output_dir}")

if __name__ == "__main__":
    main()
