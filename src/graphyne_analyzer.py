#!/usr/bin/env python3
"""
Gamma Graphyne Structure Analyzer

This script reads an XYZ file containing carbon atoms and analyzes the structure
to identify a connected gamma graphyne network. It removes heteroatoms, dangling 
and isolated atoms, detects six-membered carbon rings (benzene rings), identifies 
triple bonds based on distance and geometry, and classifies the remaining bonds 
as single bonds.

The cleaned atomic coordinates along with the connectivity information are written
to an output file named graphyne_connectivity.xyz.

"""

import sys
import os
import argparse
from itertools import combinations
import numpy as np

# --------------------- PARAMETERS --------------------- #
C_C_SINGLE_MAX = 1.60      # Maximum distance for any C-C bond
TRIPLE_BOND_MAX = 1.28     # Bonds shorter than this are considered triple
ANGLE_TOL = 20.0           # Tolerance for linear (triple bond) geometry

# --------------------- HELPER FUNCTIONS --------------------- #
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def compute_angle(a, b, c):
    """Compute angle in degrees with b as central atom."""
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle_deg = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return angle_deg


# --------------------- FILE INPUT --------------------- #
def read_xyz(filepath):
    """Read XYZ file and extract only carbon atoms."""
    if not os.path.isfile(filepath):
        print(f"ERROR: File not found -> {filepath}")
        sys.exit(1)

    with open(filepath) as f:
        lines = f.readlines()

    expected_n = int(lines[0].strip())
    atoms = []
    coords = []
    carbon_symbol = None

    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        if elem not in ["C", "6"]:
            continue
        xyz = parts[1:4]
        if not all(is_float(x) for x in xyz):
            continue

        if carbon_symbol is None:
            carbon_symbol = elem
        elif elem != carbon_symbol:
            print("ERROR: Mixed carbon notation (C and 6) not allowed.")
            sys.exit(1)

        atoms.append("C")
        coords.append([float(x) for x in xyz])

    coords_array = np.array(coords, dtype=float)
    return atoms, coords_array, expected_n, len(atoms)


# --------------------- GRAPH CONSTRUCTION --------------------- #
def build_bonds(coords):
    """Build carbon-carbon bond list using distance criterion."""
    bonds = []
    n = len(coords)
    for i, j in combinations(range(n), 2):
        distance = np.linalg.norm(coords[i] - coords[j])
        if distance < C_C_SINGLE_MAX:
            bonds.append((i, j, distance))
    return bonds


def build_adjacency_list(bonds, n_atoms):
    """Convert bond list to adjacency list."""
    adj = {i: [] for i in range(n_atoms)}
    for i, j, _ in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


# --------------------- STRUCTURE CLEANING --------------------- #
def remove_dangling_atoms(atoms, coords, bonds):
    """Iteratively remove isolated and terminal atoms."""
    removed_total = 0
    while True:
        n = len(coords)
        adj = build_adjacency_list(bonds, n)
        to_remove = [i for i in range(n) if len(adj[i]) <= 1]

        if not to_remove:
            break

        removed_total += len(to_remove)
        keep = [i for i in range(n) if i not in to_remove]
        index_map = {old: new for new, old in enumerate(keep)}

        coords = coords[keep]
        atoms = [atoms[i] for i in keep]

        new_bonds = []
        for i, j, d in bonds:
            if i in index_map and j in index_map:
                new_bonds.append((index_map[i], index_map[j], d))
        bonds = new_bonds

    return atoms, coords, bonds, removed_total


# --------------------- STRUCTURAL ANALYSIS --------------------- #
def find_rings(adj, ring_size=6):
    """Detect all unique rings of specified size."""
    rings = set()

    def dfs(path, start):
        if len(path) > ring_size:
            return
        cur = path[-1]
        for nb in adj[cur]:
            if nb == start and len(path) == ring_size:
                rings.add(tuple(sorted(path)))
            elif nb not in path:
                dfs(path + [nb], start)

    for i in adj:
        dfs([i], i)

    return [list(r) for r in rings]


def find_triple_bonds(bonds, adj, coords):
    """Identify triple bonds using distance and linearity."""
    triple_bonds = []
    for i, j, dist in bonds:
        if dist >= TRIPLE_BOND_MAX:
            continue

        is_linear = False
        for k in adj[i]:
            if k != j:
                angle = compute_angle(coords[k], coords[i], coords[j])
                if abs(angle - 180) < ANGLE_TOL:
                    is_linear = True
                    break
        if not is_linear:
            for k in adj[j]:
                if k != i:
                    angle = compute_angle(coords[i], coords[j], coords[k])
                    if abs(angle - 180) < ANGLE_TOL:
                        is_linear = True
                        break

        if is_linear:
            triple_bonds.append(tuple(sorted((i, j))))

    return list(set(triple_bonds))


def find_single_bonds(bonds, rings, triple_bonds):
    """Identify single bonds (not in rings and not triple)."""
    triple_set = set(triple_bonds)
    ring_atoms = set()
    for ring in rings:
        ring_atoms.update(ring)

    single_bonds = []
    for i, j, _ in bonds:
        pair = tuple(sorted((i, j)))
        if pair in triple_set:
            continue
        if i in ring_atoms and j in ring_atoms:
            continue
        single_bonds.append(pair)
    return list(set(single_bonds))


# --------------------- CLI HANDLING --------------------- #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Gamma Graphyne Structure Analyzer")
    parser.add_argument("-n", "--name", dest="input_file", help="Path to XYZ structure file")
    parser.add_argument("-c", "--cpu", dest="n_cpu", type=int, default=1,
                        help="Number of CPU cores")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=None,
                        help="Output directory for graphyne_connectivity.xyz")
    return parser.parse_args()


# --------------------- MAIN WORKFLOW --------------------- #
def main(input_file=None, n_cpu=1, output_dir=None):
    """
    Main workflow.
    Can be called programmatically or from CLI.
    """
    # --------------------- INPUT HANDLING --------------------- #
    if input_file is None:
        args = parse_arguments()
        input_file = args.input_file
        n_cpu = max(1, args.n_cpu)
        if args.output_dir:
            output_dir = args.output_dir

        if input_file is None:
            # Interactive mode
            print("Running in interactive mode\n")
            input_file = input("Enter structure file (path or filename): ").strip()
            cpu_input = input("Enter number of CPU cores [default: 1]: ").strip()
            n_cpu = max(1, int(cpu_input)) if cpu_input else 1

    # --------------------- WORKFLOW --------------------- #
    atoms, coords, expected_n, n_read = read_xyz(input_file)
    print(f"Number of atoms in input file: {expected_n}")
    print(f"Number of carbon atoms successfully read: {n_read}\n")

    bonds = build_bonds(coords)
    atoms, coords, bonds, removed_dangling = remove_dangling_atoms(atoms, coords, bonds)

    final_n = len(atoms)
    total_removed = expected_n - final_n
    print(f"Total number of removed atoms: {total_removed} "
          f"({removed_dangling} isolated or terminal carbon atoms)")
    print(f"Final number of carbon atoms in cleaned structure: {final_n}\n")

    adj = build_adjacency_list(bonds, final_n)
    rings = find_rings(adj)
    triple_bonds = find_triple_bonds(bonds, adj, coords)
    single_bonds = find_single_bonds(bonds, rings, triple_bonds)

    print("Analysis results for the cleaned structure:")
    print(f" - Number of benzene rings detected: {len(rings)}")
    print(f" - Number of triple bonds detected: {len(triple_bonds)}")
    print(f" - Number of single bonds detected: {len(single_bonds)}")

    # --------------------- OUTPUT --------------------- #
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file)) or "."

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "graphyne_connectivity.xyz")

    with open(output_file, "w") as f:
        f.write(f"{final_n}\n")
        f.write("Cleaned graphyne structure with connectivity information\n")
        for a, xyz in zip(atoms, coords):
            f.write(f"C {xyz[0]:15.10f} {xyz[1]:15.10f} {xyz[2]:15.10f}\n")

        f.write("\n# Connectivity Information\n")
        for i, r in enumerate(rings, 1):
            f.write(f"RING{i}: {' '.join(map(str, r))}\n")
        for i, (a, b) in enumerate(triple_bonds, 1):
            f.write(f"TRIPLE{i}: {a} {b}\n")
        for i, (a, b) in enumerate(single_bonds, 1):
            f.write(f"SINGLE{i}: {a} {b}\n")

    print(f"\nCleaned graphyne successfully written to: {output_file}")
    return output_file


# --------------------- ENTRY POINT --------------------- #
if __name__ == "__main__":
    main()
