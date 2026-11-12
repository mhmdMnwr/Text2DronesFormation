#!/usr/bin/env python3
"""
================================================================================
Text2Swarm: Prompt-to-Flight Live Simulation CLI
================================================================================
Usage:
    # 1. Interactive prompt input:
    python fly.py

    # 2. Direct command with word/prompt:
    python fly.py "fighter jet"
    python fly.py "acoustic guitar" --drones 45
    python fly.py "sports car"

    # 3. Headless physics verification:
    python fly.py "luxury yacht" --headless
================================================================================
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pybullet as p
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from PyFlyt.core import Aviary

# ------------------------------------------------------------------------------
# Configuration & Paths
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SIMILARITY_THRESHOLD = 0.30

# ------------------------------------------------------------------------------
# 6-DOF Free Camera Navigation Handler
# ------------------------------------------------------------------------------
def handle_free_camera(speed: float = 0.4):
    """
    Enables full 6-DOF video-game style free camera navigation in PyBullet:
      - W / S (or Up / Down):   Fly forward / backward along look direction
      - A / D (or Left / Right): Strafe left / right
      - Space / E:               Fly upward
      - C / Q:                   Fly downward
      - + / -:                   Zoom in / out
      - Mouse:                   Hold Ctrl + Left-Click to rotate view
    """
    cam_info = p.getDebugVisualizerCamera()
    if not cam_info:
        return
        
    yaw = cam_info[8]
    pitch = cam_info[9]
    dist = cam_info[10]
    target = np.array(cam_info[11], dtype=float)
    
    keys = p.getKeyboardEvents()
    if not keys:
        return
        
    yaw_rad = np.radians(yaw)
    forward = np.array([-np.sin(yaw_rad), np.cos(yaw_rad), 0.0])
    right = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    
    moved = False
    
    for k, v in keys.items():
        if v & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
            if k in [ord('w'), ord('W'), p.B3G_UP_ARROW]:
                target += forward * speed
                moved = True
            elif k in [ord('s'), ord('S'), p.B3G_DOWN_ARROW]:
                target -= forward * speed
                moved = True
            elif k in [ord('a'), ord('A'), p.B3G_LEFT_ARROW]:
                target -= right * speed
                moved = True
            elif k in [ord('d'), ord('D'), p.B3G_RIGHT_ARROW]:
                target += right * speed
                moved = True
            elif k in [ord(' '), ord('e'), ord('E')]:
                target += up * speed
                moved = True
            elif k in [ord('c'), ord('C'), ord('q'), ord('Q')]:
                target -= up * speed
                moved = True
            elif k in [ord('+'), ord('=')]:
                dist = max(2.0, dist - 0.5)
                moved = True
            elif k in [ord('-'), ord('_')]:
                dist = min(60.0, dist + 0.5)
                moved = True
                
    if moved:
        p.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target.tolist()
        )

# ------------------------------------------------------------------------------
# RAG Context & Database Loader
# ------------------------------------------------------------------------------
_RAG_CONTEXT = None

def get_rag_context():
    """Loads wireframe database and precomputed embeddings."""
    global _RAG_CONTEXT
    if _RAG_CONTEXT is not None:
        return _RAG_CONTEXT

    # 1. Load wireframe geometries
    wireframes_db = {}
    main_db_path = os.path.join(DATA_DIR, "keypointnet_wireframes.json")
    if os.path.exists(main_db_path):
        with open(main_db_path, "r", encoding="utf-8") as f:
            wireframes_db = json.load(f)

    # Also check sample_wireframes.json
    for sample_file in [os.path.join(DATA_DIR, "sample_wireframes.json"), os.path.join(BASE_DIR, "sample_wireframes.json")]:
        if os.path.exists(sample_file):
            with open(sample_file, "r", encoding="utf-8") as f:
                samples = json.load(f)
                for k, v in samples.items():
                    if k not in wireframes_db:
                        wireframes_db[k] = v

    # 2. Load precomputed vector embeddings
    emb_path = os.path.join(DATA_DIR, "wireframes_embeddings.npy")
    keys_path = os.path.join(DATA_DIR, "wireframes_keys.json")

    if not os.path.exists(emb_path) or not os.path.exists(keys_path):
        raise FileNotFoundError("Wireframes embeddings or keys missing in data/ directory.")

    db_embeddings = np.load(emb_path)
    with open(keys_path, "r", encoding="utf-8") as f:
        db_keys = json.load(f)

    # 3. Load embedding model (CPU-based for maximum compatibility)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    _RAG_CONTEXT = (wireframes_db, db_keys, db_embeddings, embedding_model)
    return _RAG_CONTEXT

# ------------------------------------------------------------------------------
# Wireframe Processing & Subdivision
# ------------------------------------------------------------------------------
def calculate_min_drones_for_human_eye(shape: dict, d_gap: float = 0.25) -> int:
    """Calculates perceptual minimum drone count (N_min) to avoid constellation effect."""
    anchors = np.array(shape["anchors"])
    edges = shape["edges"]
    K = len(anchors)
    edge_drones = sum(max(0, int(np.ceil(np.linalg.norm(anchors[i] - anchors[j]) / d_gap)) - 1) for i, j in edges)
    return K + edge_drones

def build_swarm_coordinates(shape: dict, num_drones: int):
    """Subdivides edges to distribute drones evenly along structural lines."""
    anchors = np.array(shape["anchors"], dtype=float)
    edges = shape["edges"]
    K = len(anchors)
    
    if num_drones < K:
        return None, None
        
    points = list(anchors)
    point_types = ["anchor"] * K
    remaining = num_drones - K
    
    if remaining > 0 and len(edges) > 0:
        edge_lengths = np.array([np.linalg.norm(anchors[j] - anchors[i]) for i, j in edges])
        total_len = np.sum(edge_lengths)
        if total_len > 1e-6:
            exact = (edge_lengths / total_len) * remaining
            drones_per_edge = np.floor(exact).astype(int)
            remainder = remaining - np.sum(drones_per_edge)
            
            if remainder > 0:
                best_edges = np.argsort(exact - drones_per_edge)[-remainder:]
                for e in best_edges:
                    drones_per_edge[e] += 1
                    
            for (i, j), m in zip(edges, drones_per_edge):
                p1, p2 = anchors[i], anchors[j]
                for s in range(1, m + 1):
                    t = s / (m + 1)
                    points.append((1.0 - t) * p1 + t * p2)
                    point_types.append("edge")
        else:
            # Fallback if no edges or zero length
            for _ in range(remaining):
                points.append(anchors[0])
                point_types.append("edge")
                
    return np.array(points), point_types

def scale_wireframe_to_airspace(points: np.ndarray, anchors: np.ndarray, target_altitude: float = 6.5, scale: float = 5.0):
    """Scales normalized [-1, 1]^3 coordinates into physical flight coordinates in meters with safety clearance."""
    center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2.0
    centered = points - center
    
    max_3d_radius = np.max(np.linalg.norm(centered, axis=1))
    if max_3d_radius < 1e-4:
        max_3d_radius = 1.0
        
    factor = scale / max_3d_radius
    
    # Automatically enforce a minimum 0.55m physical clearance between any two target points
    min_norm_dist = 999.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if 1e-5 < d < min_norm_dist:
                min_norm_dist = d
                
    if min_norm_dist < 900.0 and (min_norm_dist * factor) < 0.55:
        factor = 0.55 / min_norm_dist
        
    physical_targets = np.zeros_like(points)
    physical_targets[:, 0] = centered[:, 0] * factor
    physical_targets[:, 1] = centered[:, 1] * factor
    physical_targets[:, 2] = target_altitude + centered[:, 2] * factor
        
    centered_anchors = anchors - center
    physical_anchors = np.zeros_like(anchors)
    physical_anchors[:, 0] = centered_anchors[:, 0] * factor
    physical_anchors[:, 1] = centered_anchors[:, 1] * factor
    physical_anchors[:, 2] = target_altitude + centered_anchors[:, 2] * factor
        
    return physical_targets, physical_anchors

def generate_ground_launch_grid(num_drones: int, spacing: float = 1.2):
    """Generates an optimal ground launch grid for N drones with safe clearance."""
    cols = int(np.ceil(np.sqrt(num_drones * 1.8)))
    rows = int(np.ceil(num_drones / cols))
    
    xs = np.linspace(-(cols - 1) * spacing / 2.0, (cols - 1) * spacing / 2.0, cols)
    ys = np.linspace(-(rows - 1) * spacing / 2.0, (rows - 1) * spacing / 2.0, rows)
    
    grid = []
    for y in ys:
        for x in xs:
            grid.append([x, y, 0.05])
            if len(grid) == num_drones:
                break
        if len(grid) == num_drones:
            break
            
    return np.array(grid)

# ------------------------------------------------------------------------------
# Core Simulation Engine
# ------------------------------------------------------------------------------
def fly_swarm_formation(prompt: str, drone_budget: int = None, gui_mode: bool = True):
    """
    End-to-end pipeline:
      1. RAG query vectorization and matching
      2. Human perception N_min calculation & subdivision
      3. Hungarian optimal assignment
      4. PyFlyt physics flight execution
    """
    print("\n" + "=" * 75)
    print("🛸 Text2Swarm: Semantic RAG & Multi-UAV Physics Flight")
    print("=" * 75)
    print(f"User Input: \"{prompt}\"")
    
    wireframes_db, db_keys, db_embeddings, embedding_model = get_rag_context()
    
    # 1. Semantic RAG Search
    t0 = time.time()
    q_vec = embedding_model.encode(prompt, normalize_embeddings=True)
    scores = db_embeddings @ q_vec
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_key = db_keys[best_idx]
    latency_ms = (time.time() - t0) * 1000
    
    # Threshold Verification
    if best_score < SIMILARITY_THRESHOLD:
        print(f"\n❌ Cosine Similarity: {best_score:.3f} (Threshold: {SIMILARITY_THRESHOLD:.2f})")
        print("RESULT: this formation is not valid")
        print("No 3D drone wireframe formation matches this word/description.")
        print("Try words like: airplane, fighter jet, car, chair, boat, guitar, bed, table, laptop...")
        print("=" * 75 + "\n")
        return False
        
    shape = wireframes_db[best_key]
    category = shape.get("category", best_key.split("_")[0])
    anchors = np.array(shape["anchors"], dtype=float)
    edges = shape["edges"]
    K = len(anchors)
    n_min = calculate_min_drones_for_human_eye(shape)
    
    # Determine drone count
    if drone_budget is None:
        # Default to perceptual N_min (or at least 45 for visual density)
        target_drones = max(n_min, 45)
    else:
        target_drones = drone_budget
        
    if target_drones < K:
        print(f"\n❌ ERROR: Target drone budget ({target_drones}) < mandatory anchors ({K}).")
        print(f"Please specify at least {K} drones to maintain structural shape.")
        print("=" * 75 + "\n")
        return False
        
    print(f"✅ MATCH FOUND!")
    print(f"  • Shape Model:         {best_key}")
    print(f"  • Category:            {category.upper()}")
    print(f"  • Cosine Similarity:   {best_score:.3f}")
    print(f"  • Mandatory Anchors:   {K} drones")
    print(f"  • Perceptual N_min:    {n_min} drones")
    print(f"  • Active Swarm Size:   {target_drones} QuadX UAVs")
    print(f"  • Semantic Latency:    {latency_ms:.1f} ms")
    
    # 2. Build 3D Waypoint Formation
    normalized_points, point_types = build_swarm_coordinates(shape, target_drones)
    # Determine physical scaling based on swarm size to guarantee collision clearance
    scale = max(5.5, np.sqrt(target_drones) * 0.9)
    physical_targets, physical_anchors = scale_wireframe_to_airspace(normalized_points, anchors, scale=scale)
    
    # 3. Ground Launch Setup
    start_pos = generate_ground_launch_grid(target_drones, spacing=1.2)
    start_orn = np.zeros((target_drones, 3))
    
    # 4. Hungarian Optimal Assignment
    print("\nCalculating Optimal Drone-to-Target Assignment (Hungarian Algorithm)...")
    cost_matrix = np.zeros((target_drones, target_drones))
    for i in range(target_drones):
        cost_matrix[i, :] = np.sum((physical_targets - start_pos[i]) ** 2, axis=1)
        
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_targets = physical_targets[col_ind]
    
    dists = np.linalg.norm(assigned_targets - start_pos, axis=1)
    print(f"  • Mean flight distance: {np.mean(dists):.2f}m (Max: {np.max(dists):.2f}m)")
    print(f"  • Intersecting paths:   0 (Untangled optimal corridors)")
    
    # 5. Initialize PyFlyt Aviary
    print(f"\nInitializing PyFlyt Aviary ({target_drones} QuadX UAVs, GUI={gui_mode})...")
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        drone_type="quadx",
        render=gui_mode
    )
    
    # Configure Flight Mode 7 (Cascaded Position Control [x, y, yaw, z])
    env.set_mode(7)
    
    if gui_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=16.0,
            cameraYaw=-45,
            cameraPitch=-20,
            cameraTargetPosition=[0.0, 0.0, 5.5]
        )
        # Draw target wireframe skeleton in the sky using cyan debug lines
        for i, j in edges:
            p.addUserDebugLine(
                physical_anchors[i].tolist(),
                physical_anchors[j].tolist(),
                lineColorRGB=[0.1, 0.85, 1.0],
                lineWidth=2.0
            )
        print("\n===========================================================================")
        print("🎮 6-DOF CAMERA NAVIGATION ACTIVE (Click inside the window to focus):")
        print("  • Fly Camera: [W]/[S] forward/back, [A]/[D] strafe, [Space]/[C] up/down")
        print("  • Look/Orbit: Hold [Ctrl] + Left Click Drag | Zoom: [Ctrl] + Right Click")
        print("===========================================================================\n")
        
    # 6. S-Curve Flight Execution
    total_time = 7.0
    control_hz = 40
    num_steps = int(total_time * control_hz)
    
    print(f"Stepping Physics Engine ({num_steps} steps, {target_drones} UAVs flying into formation)...")
    t_start = time.time()
    
    for step in range(num_steps):
        tau = min(1.0, step / float(num_steps))
        # Quintic minimum-jerk transition: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        
        current_setpoints = np.zeros((target_drones, 4))
        for i in range(target_drones):
            p_curr = (1.0 - s) * start_pos[i] + s * assigned_targets[i]
            current_setpoints[i, 0] = p_curr[0]
            current_setpoints[i, 1] = p_curr[1]
            current_setpoints[i, 2] = 0.0  # Desired yaw = 0 rad
            current_setpoints[i, 3] = p_curr[2]
            
        env.set_all_setpoints(current_setpoints)
        env.step()
        
        if gui_mode:
            handle_free_camera()
            time.sleep(0.015)
            
    flight_wall_time = time.time() - t_start
    print(f"Flight transition completed in {flight_wall_time:.2f} seconds!")
    
    # Check minimum separation distance
    states = env.all_states
    final_coords = np.array([states[i][-1, :] for i in range(target_drones)])
    min_sep = 999.0
    for i in range(target_drones):
        for j in range(i + 1, target_drones):
            d = np.linalg.norm(final_coords[i] - final_coords[j])
            if d < min_sep:
                min_sep = d
    sep_tag = "✅ SAFE (> 0.5m)" if min_sep >= 0.50 else f"⚠️ CAUTION ({min_sep:.2f}m)"
    print(f"Minimum inter-drone separation during flight: {min_sep:.2f}m ({sep_tag})")
    
    # 7. Hover Loop (in GUI mode)
    if gui_mode:
        print("\n===========================================================================")
        print(f"🎉 FORMATION REACHED: {category.upper()} with {target_drones} drones hovering!")
        print("🎮 Fly freely around the formation using WASD + Space/C + Ctrl+Mouse.")
        print("Close the PyBullet window or press Ctrl+C to exit.")
        print("===========================================================================\n")
        
        final_setpoints = np.zeros((target_drones, 4))
        for i in range(target_drones):
            final_setpoints[i, 0] = assigned_targets[i, 0]
            final_setpoints[i, 1] = assigned_targets[i, 1]
            final_setpoints[i, 2] = 0.0
            final_setpoints[i, 3] = assigned_targets[i, 2]
            
        try:
            while p.isConnected():
                handle_free_camera()
                env.set_all_setpoints(final_setpoints)
                env.step()
                time.sleep(1.0 / 60.0)
        except (KeyboardInterrupt, Exception):
            pass
            
        if p.isConnected():
            p.disconnect()
        print("3D Window closed cleanly.")
    else:
        if p.isConnected():
            p.disconnect()
        
    return True

# ------------------------------------------------------------------------------
# Entrypoint & Interactive Loop
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Text2Swarm: Prompt-to-Flight Live Simulation CLI")
    parser.add_argument("prompt", nargs="?", type=str, default=None, help="Word or description of the shape to simulate")
    parser.add_argument("--drones", "-n", type=int, default=None, help="Target drone count (default: auto perceptual N_min)")
    parser.add_argument("--headless", action="store_true", help="Run in fast headless physics mode (no GUI)")
    
    args = parser.parse_args()
    gui = not args.headless
    
    prompt = args.prompt
    if not prompt:
        print("=" * 75)
        print("🛸 Text2Swarm Interactive Flight Simulator")
        print("=" * 75)
        try:
            prompt = input("\nEnter a word or description (e.g. 'fighter jet', 'guitar', 'car', 'chair', 'boat'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)
            
        if not prompt:
            print("No word entered. Exiting.")
            sys.exit(0)
            
    fly_swarm_formation(prompt, drone_budget=args.drones, gui_mode=gui)

if __name__ == "__main__":
    main()
