#!/usr/bin/env python3
"""
Hydrogen Addition to Unique Clusters (Final Stage)

Takes unique graphyne clusters from "unique_clusters/" and adds hydrogen atoms
to under-coordinated sp2 carbon atoms.

Output naming: H_unique_XXXXXXXXX_CnHm.xyz
"""

import os
import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import argparse

# --------------------- PARAMETERS --------------------- #
NEIGHBOR_CUTOFF = 1.60
C_H_BOND_LENGTH = 1.09

# --------------------- CLI HANDLING --------------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hydrogen addition step of Graphyne pipeline"
    )
    parser.add_argument(
        "-i", "--input-dir",
        dest="input_dir",
        default="unique_clusters",
        help="Input directory containing unique_*.xyz files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        dest="output_dir",
        default=None,
        help="Base output directory (H-structures will be written here)"
    )
    parser.add_argument(
        "-c", "--cpu",
        dest="n_cpu",
        type=int,
        default=1,
        help="Number of CPU cores"
    )
    return parser.parse_args()


# --------------------- HELPER FUNCTIONS --------------------- #
def get_worker_count(n_cpu):
    """Determine safe number of worker processes."""
    cpu_count = os.cpu_count() or 4
    n_cpu = max(1, n_cpu)
    return max(1, min(8, min(cpu_count - 2, n_cpu)))


# --------------------- FILE INPUT --------------------- #
def read_xyz(filename):
    """Read XYZ file."""
    with open(filename, "r") as f:
        lines = [line.rstrip() for line in f]

    n_atoms = int(lines[0])
    title = lines[1]

    symbols = []
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))

    return symbols, np.array(coords), title


# --------------------- GEOMETRY --------------------- #
def build_neighbors(coords):
    """Build neighbor list based on distance."""
    n = len(coords)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) < NEIGHBOR_CUTOFF:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def compute_angle(v1, v2):
    """Compute angle between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    v1 = v1 / n1
    v2 = v2 / n2
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def classify_carbon(i, coords, neighbors):
    """Classify carbon hybridization."""
    n_neighbors = len(neighbors[i])
    if n_neighbors == 1:
        return "sp"
    if n_neighbors == 2:
        j, k = neighbors[i]
        ang = compute_angle(coords[j] - coords[i], coords[k] - coords[i])
        return "sp" if ang > 160 else "sp2"
    return "sp3"


def should_add_hydrogen(symbol, carbon_type):
    return symbol == "C" and carbon_type == "sp2"


# --------------------- HYDROGEN ADDITION --------------------- #
def add_hydrogens(symbols, coords, neighbors):
    """Add hydrogens to sp2 carbons with only two neighbors."""
    new_symbols = list(symbols)
    new_coords = list(coords)

    for i, symbol in enumerate(symbols):
        if symbol != "C":
            continue

        ctype = classify_carbon(i, coords, neighbors)
        if not should_add_hydrogen(symbol, ctype):
            continue

        # Compute direction (average away from bonded neighbors)
        direction = np.zeros(3)
        for j in neighbors[i]:
            direction += (coords[i] - coords[j])

        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

        h_pos = coords[i] + C_H_BOND_LENGTH * direction

        new_symbols.append("H")
        new_coords.append(h_pos)

    return new_symbols, np.array(new_coords)


# --------------------- OUTPUT --------------------- #
def write_xyz(filename, symbols, coords):
    """Write XYZ file with added hydrogens."""
    with open(filename, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write("H-added corrected structure\n")
        for s, c in zip(symbols, coords):
            f.write(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


# --------------------- WORKER --------------------- #
def process_structure(args):
    """Worker function for multiprocessing."""
    input_file, output_dir = args
    base_name = os.path.basename(input_file)
    identifier = base_name.replace("unique_", "").replace(".xyz", "")

    symbols, coords, _ = read_xyz(input_file)
    neighbors = build_neighbors(coords)
    new_symbols, new_coords = add_hydrogens(symbols, coords, neighbors)

    nC = sum(1 for s in new_symbols if s == "C")
    nH = sum(1 for s in new_symbols if s == "H")

    output_name = os.path.join(
        output_dir,
        f"H_unique_{identifier}_C{nC}H{nH}.xyz"
    )

    write_xyz(output_name, new_symbols, new_coords)
    return output_name


# --------------------- MAIN --------------------- #
def main(input_dir="unique_clusters", output_dir=None, n_cpu=1):
    """Main function with output directory support."""
    
    if output_dir is None:
        base_dir = os.getcwd()
    else:
        base_dir = os.path.abspath(output_dir)

    # Resolve input directory (support both relative and absolute)
    if os.path.isabs(input_dir):
        input_dir_path = input_dir
    else:
        input_dir_path = os.path.join(base_dir, input_dir)

    output_h_dir = os.path.join(base_dir, "H-structures")
    os.makedirs(output_h_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir_path, "unique_*.xyz")))
    if not files:
        print(f"ERROR: No unique_*.xyz files found in {input_dir_path}")
        return

    workers = get_worker_count(n_cpu)
    print(f"Found {len(files)} unique structures.")
    print(f"Using {workers} CPU cores for hydrogen addition")
    print(f"Output directory: {output_h_dir}\n")

    tasks = [(f, output_h_dir) for f in files]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.imap_unordered(process_structure, tasks)
        for _ in tqdm(results, total=len(files), desc="Adding hydrogens"):
            pass

    print("\nHydrogen addition completed successfully.")
    print(f"H-added structures saved to: {output_h_dir}")


# --------------------- ENTRY POINT --------------------- #
if __name__ == "__main__":
    args = parse_arguments()
    mp.set_start_method("spawn", force=True)
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_cpu=args.n_cpu
    )
