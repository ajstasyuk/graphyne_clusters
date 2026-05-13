#!/usr/bin/env python3
"""
Second Stage: Graphyne Sheet Cluster Generator

This script takes the cleaned graphyne sheet produced by the first script
(graphyne_connectivity.xyz) and cuts it into smaller triangular clusters
based on the connectivity of benzene rings linked by triple bonds.

The workflow is as follows:
1. Read the structure and detected rings from the XYZ file.
2. Build the atomic connectivity graph.
3. Identify triple bonds and construct a ring connectivity graph.
4. Detect triangular motifs (three rings mutually connected).
5. Enumerate all connected groups of these triangles.
6. For each group, collect the corresponding atoms and write them
   as individual cluster XYZ files.

"""

import os
import multiprocessing as mp
from multiprocessing import cpu_count
from collections import deque
from itertools import combinations
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm

# --------------------- PARAMETERS --------------------- #

# Distance cutoffs used for bond perception from coordinates

# Maximum distance to consider two carbon atoms as bonded
BOND_MAX = 1.60

# Distance range used to identify carbon-carbon triple bonds
TRIPLE_MIN = 1.14
TRIPLE_MAX = 1.28

# Multiprocessing settings
CHUNKSIZE = 250         # Number of tasks per chunk sent to each worker


# --------------------- INPUT READING --------------------- #
def read_xyz_with_rings(filename):
    """
    Read an XYZ file that contains both atomic coordinates and ring
    information written by the first script.
    Returns atoms, coordinates, and list of rings (each ring is a list of atom indices).
    """
    atoms = []
    coords = []
    rings = []

    with open(filename) as f:
        lines = f.readlines()

    natoms = int(lines[0].strip())

    # Read atomic coordinates
    for line in lines[2:2 + natoms]:
        parts = line.split()
        s = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(s)
        coords.append([x, y, z])

    coords = np.array(coords, dtype=float)

    # Read RING definitions
    for line in lines[2 + natoms:]:
        if line.startswith("RING"):
            ring_atoms = list(map(int, line.split()[1:]))
            rings.append(ring_atoms)

    return atoms, coords, rings


# --------------------- GRAPH CONSTRUCTION --------------------- #
def build_atom_graph(atoms, coords):
    """
    Build an undirected graph of carbon atoms using a KDTree for efficient
    neighbor search. Edges are added between carbon atoms within BOND_MAX distance.
    """
    G = nx.Graph()
    tree = cKDTree(coords)
    for i, j in tree.query_pairs(BOND_MAX):
        if atoms[i] == "C" and atoms[j] == "C":
            dist = np.linalg.norm(coords[i] - coords[j])
            G.add_edge(i, j, dist=dist)
    return G


def detect_triple_bonds(atom_graph):
    """Identify triple bonds based on bond length."""
    triple_bonds = []
    for i, j, data in atom_graph.edges(data=True):
        dist = data.get("dist")
        if TRIPLE_MIN <= dist <= TRIPLE_MAX:
            triple_bonds.append((i, j))
    return triple_bonds


def build_triple_graph(triple_bonds):
    """Create graph containing only triple bonds."""
    TG = nx.Graph()
    TG.add_edges_from(triple_bonds)
    return TG


def build_ring_graph(rings, atom_graph, triple_bonds):
    """Build graph where nodes are rings and edges are triple-bond connections."""
    RG = nx.Graph()
    for i in range(len(rings)):
        RG.add_node(i)

    atom_to_ring = {atom: rid for rid, ring in enumerate(rings) for atom in ring}
    triple_graph = build_triple_graph(triple_bonds)

    def reachable_rings(start_atom):
        """
        Perform a breadth-first search starting from an atom to find
        all rings reachable without crossing triple bonds.
        """
        visited = {start_atom}
        queue = deque([start_atom])
        found_rings = set()

        while queue:
            node = queue.popleft()

            # If we reach an atom in a ring, record it
            if node in atom_to_ring:
                found_rings.add(atom_to_ring[node])
                continue

            # Traverse single bonds
            for nb in atom_graph.neighbors(node):
                if triple_graph.has_edge(node, nb):
                    continue
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

            # Also traverse along triple bonds
            if node in triple_graph:
                for nb in triple_graph.neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

        return found_rings

    # Connect rings that are linked by triple bonds
    for a, b in triple_bonds:
        rings_a = reachable_rings(a)
        rings_b = reachable_rings(b)
        for ra in rings_a:
            for rb in rings_b:
                if ra != rb:
                    if RG.has_edge(ra, rb):
                        RG[ra][rb]["triples"].append((a, b))
                    else:
                        RG.add_edge(ra, rb, triples=[(a, b)])

    return RG


def find_triangles(ring_graph):
    """
    Find all sets of three rings that are mutually connected (triangles)
    in the ring connectivity graph.
    """
    triangles = []
    for clique in nx.enumerate_all_cliques(ring_graph):
        if len(clique) == 3:
            triangles.append(tuple(sorted(clique)))
    return sorted(set(triangles))


def build_triangle_graph(triangles):
    """
    Build a graph where nodes are triangular motifs, and edges connect
    triangles that share at least one ring.
    """
    TG = nx.Graph()
    for i in range(len(triangles)):
        TG.add_node(i)

    for i, t1 in enumerate(triangles):
        for j, t2 in enumerate(triangles[i + 1:], start=i + 1):
            if set(t1) & set(t2):
                TG.add_edge(i, j)
    return TG


# --------------------- CLUSTER ENUMERATION --------------------- #
def enumerate_all_triangle_clusters(triangle_graph, min_size=2):
    """Enumerate all connected subgraphs of triangles."""
    clusters = []
    nodes = list(triangle_graph.nodes)
    n = len(nodes)

    for size in range(min_size, n + 1):
        for combo in combinations(nodes, size):
            subgraph = triangle_graph.subgraph(combo)
            if nx.is_connected(subgraph):
                clusters.append(tuple(sorted(combo)))
    return sorted(clusters)


# --------------------- ATOM COLLECTION --------------------- #
def collect_atoms_for_triangle(triangle, rings, atom_graph, ring_graph, triple_bonds):
    """Collect all atoms belonging to a triangular motif."""
    triple_graph = build_triple_graph(triple_bonds)
    atom_set = set()

    # Add atoms from the three rings
    for ring_id in triangle:
        atom_set.update(rings[ring_id])

    # Add connecting triple bonds and linker atoms
    pairs = [(triangle[0], triangle[1]),
             (triangle[1], triangle[2]),
             (triangle[0], triangle[2])]

    for ra, rb in pairs:
        if ring_graph.has_edge(ra, rb):
            for a, b in ring_graph[ra][rb]["triples"]:
                atom_set.add(a)
                atom_set.add(b)
                atom_set.update(atoms_to_ring(a, atom_graph, rings, triple_graph))
                atom_set.update(atoms_to_ring(b, atom_graph, rings, triple_graph))

    return sorted(atom_set)


def atoms_to_ring(start, atom_graph, rings, triple_graph):
    """Collect atoms reachable without crossing triple bonds."""
    ring_atoms = {a for ring in rings for a in ring}
    visited = {start}
    queue = deque([start])
    collected = {start}

    while queue:
        node = queue.popleft()
        if node in ring_atoms:
            continue
        for nb in atom_graph.neighbors(node):
            if triple_graph.has_edge(node, nb):
                continue
            if nb not in visited:
                visited.add(nb)
                collected.add(nb)
                queue.append(nb)

        if node in triple_graph:
            for nb in triple_graph.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    collected.add(nb)
                    queue.append(nb)
    return collected


# --------------------- PARALLEL WRITING --------------------- #
def init_worker(triangles_, rings_, atoms_, coords_, atom_graph_, ring_graph_, 
                triple_bonds_, outdir_):
    global triangles, rings, atoms, coords, atom_graph, ring_graph, triple_bonds, outdir
    triangles = triangles_
    rings = rings_
    atoms = atoms_
    coords = coords_
    atom_graph = atom_graph_
    ring_graph = ring_graph_
    triple_bonds = triple_bonds_
    outdir = outdir_


def write_single_cluster(args):
    """Worker function to write one cluster."""
    cluster_id, triangle_indices = args
    atom_set = set()
    for tid in triangle_indices:
        atom_set.update(
            collect_atoms_for_triangle(
                triangles[tid], rings, atom_graph, ring_graph, triple_bonds
            )
        )

    atom_list = sorted(atom_set)
    filename = os.path.join(outdir, f"cluster_{cluster_id:09d}.xyz")

    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\n")
        f.write(f"Cluster {cluster_id} triangles {sorted(triangle_indices)}\n")
        for a in atom_list:
            x, y, z = coords[a]
            f.write(f"{atoms[a]} {x:.6f} {y:.6f} {z:.6f}\n")

    return cluster_id


def write_all_triangle_clusters(triangles, clusters, rings, atoms, coords,
                                atom_graph, ring_graph, triple_bonds,
                                n_cpu=1, output_dir="triangle_clusters_all"):
    """Write all clusters to disk using multiprocessing."""
    outdir_full = os.path.abspath(output_dir)
    os.makedirs(outdir_full, exist_ok=True)

    total = len(clusters)
    print(f"Total number of triangle clusters to save: {total}")
    print(f"Using {n_cpu} CPU cores for writing clusters\n")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_cpu, initializer=init_worker,
                  initargs=(triangles, rings, atoms, coords, atom_graph,
                            ring_graph, triple_bonds, outdir_full)) as pool:
        
        stream = zip(range(total), clusters)
        list(tqdm(
            pool.imap_unordered(write_single_cluster, stream, chunksize=CHUNKSIZE),
            total=total,
            desc="Saving clusters",
            unit="cluster"
        ))

    print(f"All triangle clusters written successfully to: {outdir_full}")


# --------------------- MAIN WORKFLOW --------------------- #
def generate_all_triangle_clusters(input_filename,
                                   min_cluster_size=2,
                                   n_cpu=1,
                                   output_dir=None):
    """
    Main function that coordinates the entire process of cutting a large                        
    graphyne sheet into smaller triangular clusters.                                            
    
    Parameters:
        output_dir: Base output directory. Clusters will be saved in 
                    {output_dir}/triangle_clusters_all/
    """
    if output_dir is None:
        output_dir = "."

    triangle_output_dir = os.path.join(output_dir, "triangle_clusters_all")

    print("Reading input structure and ring information...")
    atoms, coords, rings = read_xyz_with_rings(input_filename)

    print("Building atomic connectivity graph...")
    atom_graph = build_atom_graph(atoms, coords)

    print("Detecting triple bonds...")
    triple_bonds = detect_triple_bonds(atom_graph)

    print("Building ring connectivity graph...")
    ring_graph = build_ring_graph(rings, atom_graph, triple_bonds)

    print("Finding triangular motifs...")
    triangles = find_triangles(ring_graph)

    print("Building triangle connectivity graph...")
    triangle_graph = build_triangle_graph(triangles)

    print(f"Enumerating all connected triangle clusters (min size = {min_cluster_size})...")
    clusters = enumerate_all_triangle_clusters(triangle_graph, min_size=min_cluster_size)

    write_all_triangle_clusters(
        triangles=triangles,
        clusters=clusters,
        rings=rings,
        atoms=atoms,
        coords=coords,
        atom_graph=atom_graph,
        ring_graph=ring_graph,
        triple_bonds=triple_bonds,
        n_cpu=n_cpu,
        output_dir=triangle_output_dir
    )

    print(f"Triangle cluster generation completed. Output: {triangle_output_dir}")
    return triangles, clusters, ring_graph, triangle_graph


# --------------------- ENTRY POINT --------------------- #
if __name__ == "__main__":
    # Standalone mode (for testing)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input connectivity file")
    parser.add_argument("-o", "--output-dir", default=None, help="Base output directory")
    parser.add_argument("-c", "--cpu", type=int, default=1)
    args = parser.parse_args()

    generate_all_triangle_clusters(
        input_filename=args.input,
        n_cpu=args.cpu,
        output_dir=args.output_dir
    )
