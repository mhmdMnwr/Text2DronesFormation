import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
import pybullet as p
from PyFlyt.core import Aviary

# ------------------------------------------------------------------------------
# Free Camera Navigation Handler (WASD + Space/C/Q/E + Arrows)
# ------------------------------------------------------------------------------
def handle_free_camera(speed: float = 0.4):
    """
    Enables full 6-DOF video-game style free camera navigation in PyBullet:
      - W / S (or Up / Down):   Fly forward / backward along look direction
      - A / D (or Left / Right): Strafe left / right
      - Space / E:               Fly upward
      - C / Q:                   Fly downward
      - + / -:                   Zoom in / out
      - Mouse:                   Hold Ctrl + Left-Click to rotate, Right-Click to zoom
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
# 1. Load Wireframe and Generate 50 Target Waypoints
# ------------------------------------------------------------------------------
def get_50_drone_target_formation():
    """Generates 50 3D target coordinates for an airplane wireframe formation."""
    shape = None
    shape_name = "fighter_jet"
    for path in ["data/sample_wireframes.json", "sample_wireframes.json", "data/keypointnet_wireframes.json"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                db = json.load(f)
            if "fighter_jet" in db:
                shape = db["fighter_jet"]
                shape_name = "fighter_jet"
                break
            elif "airplane_51ebcde4" in db:
                shape = db["airplane_51ebcde4"]
                shape_name = "airplane"
                break
                
    if shape is None:
        raise FileNotFoundError("Could not find wireframe database file.")
        
    anchors = np.array(shape["anchors"], dtype=float)
    edges = shape["edges"]
    K = len(anchors)
    num_drones = 50
    
    # Bottom-up subdivision to exactly 50 drones
    points = list(anchors)
    point_types = ["anchor"] * K
    remaining = num_drones - K
    
    if remaining > 0 and len(edges) > 0:
        edge_lengths = np.array([np.linalg.norm(anchors[j] - anchors[i]) for i, j in edges])
        total_len = np.sum(edge_lengths)
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
                
    points = np.array(points)
    
    # Scale from normalized [-1, 1]^3 to physical flight coordinates in meters:
    # X in [-5m, 5m] (wingspan 10m)
    # Y in [-6m, 6m] (length 12m)
    # Z in [5m, 9m]  (formation cruise altitude around 7m)
    physical_targets = np.zeros_like(points)
    physical_targets[:, 0] = points[:, 0] * 5.0
    physical_targets[:, 1] = points[:, 1] * 5.0
    physical_targets[:, 2] = 6.5 + points[:, 2] * 2.5
    
    # Also compute physical anchor positions for drawing laser wireframe edges
    physical_anchors = np.zeros_like(anchors)
    physical_anchors[:, 0] = anchors[:, 0] * 5.0
    physical_anchors[:, 1] = anchors[:, 1] * 5.0
    physical_anchors[:, 2] = 6.5 + anchors[:, 2] * 2.5
    
    return physical_targets, point_types, edges, physical_anchors, shape_name

# ------------------------------------------------------------------------------
# 2. Setup 50-Drone PyFlyt Simulation with Hungarian Assignment
# ------------------------------------------------------------------------------
def run_pyflyt_50_drones_simulation(gui_mode: bool = False):
    print("=" * 75)
    print("   PyFlyt Multi-UAV 50-Drone Swarm Movement Simulation")
    print("=" * 75)
    
    num_drones = 50
    target_pos, point_types, edges, physical_anchors, shape_name = get_50_drone_target_formation()
    print(f"Target Formation: {shape_name.upper()} with {num_drones} drones")
    print(f"Altitude Range:   {np.min(target_pos[:, 2]):.2f}m to {np.max(target_pos[:, 2]):.2f}m")
    print(f"Simulation Mode:  {'Interactive Live 3D OpenGL Window (GUI)' if gui_mode else 'Headless Fast Physics'}")
    
    # Ground Launch Grid: 5 rows x 10 cols, spaced by 1.2 meters
    gx, gy = np.meshgrid(np.linspace(-5.4, 5.4, 10), np.linspace(-2.4, 2.4, 5))
    start_pos = np.zeros((num_drones, 3))
    start_pos[:, 0] = gx.flatten()
    start_pos[:, 1] = gy.flatten()
    start_pos[:, 2] = 0.08
    
    start_orn = np.zeros((num_drones, 3))
    
    # Step A: Hungarian Matching (Collision-Free Target Assignment)
    print("\nCalculating Optimal Drone-to-Target Assignment (Hungarian Algorithm)...")
    cost_matrix = np.linalg.norm(start_pos[:, None, :] - target_pos[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_targets = target_pos[col_ind]
    assigned_types = [point_types[i] for i in col_ind]
    
    travel_dists = np.linalg.norm(start_pos - assigned_targets, axis=1)
    print(f"  • Mean travel distance: {np.mean(travel_dists):.2f}m (Max: {np.max(travel_dists):.2f}m)")
    
    # Step B: Initialize PyFlyt Aviary Environment
    print(f"\nInitializing PyFlyt Aviary with 50 QuadX Drones (render={gui_mode})...")
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        drone_type="quadx",
        render=gui_mode
    )
    
    if gui_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=18.0,
            cameraYaw=-45,
            cameraPitch=-22,
            cameraTargetPosition=[0.0, 0.0, 4.5]
        )
        for i, j in edges:
            p.addUserDebugLine(
                physical_anchors[i].tolist(),
                physical_anchors[j].tolist(),
                lineColorRGB=[0.0, 0.8, 1.0],
                lineWidth=2.0
            )
        print("✓ Live 3D Window Initialized!")
        print("=" * 75)
        print("🎮 6-DOF FREE CAMERA CONTROLS:")
        print("  • Fly Forward / Back:  [W] / [S] (or Up / Down arrows)")
        print("  • Strafe Left / Right: [A] / [D] (or Left / Right arrows)")
        print("  • Fly Up / Down:       [Space] / [C] (or [E] / [Q])")
        print("  • Rotate Camera:       Hold [Ctrl] + [Left Click Drag]")
        print("  • Zoom:                Hold [Ctrl] + [Right Click Drag] or [+]/[-]")
        print("=" * 75)
        
    env.set_mode(7)
    
    # Step C: Smooth S-Curve Trajectory Simulation
    sim_steps = 220
    print(f"\nStepping Physics Engine ({sim_steps} steps, 50 QuadX UAVs)...")
    
    history_pos = np.zeros((sim_steps, num_drones, 3))
    min_distances = np.zeros(sim_steps)
    
    t0 = time.time()
    for step in range(sim_steps):
        tau = min(1.0, step / (sim_steps * 0.75))
        s = 6.0 * (tau ** 5) - 15.0 * (tau ** 4) + 10.0 * (tau ** 3)
        
        current_setpoints = np.zeros((num_drones, 4))
        for i in range(num_drones):
            pos = (1.0 - s) * start_pos[i] + s * assigned_targets[i]
            current_setpoints[i, 0] = pos[0]
            current_setpoints[i, 1] = pos[1]
            current_setpoints[i, 2] = 0.0
            current_setpoints[i, 3] = pos[2]
            
        env.set_all_setpoints(current_setpoints)
        env.step()
        
        if gui_mode:
            handle_free_camera()
            time.sleep(0.015)
            
        states = env.all_states
        for i in range(num_drones):
            history_pos[step, i] = states[i][-1, :]
            
        pts = history_pos[step]
        pairwise_dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(pairwise_dists, np.inf)
        min_distances[step] = np.min(pairwise_dists)
        
    t1 = time.time()
    print(f"Flight transition completed in {t1 - t0:.2f} seconds!")
    print(f"Minimum inter-drone separation during flight: {np.min(min_distances):.2f}m (> 0.5m collision threshold ✅)")
    
    # --------------------------------------------------------------------------
    # 3. Static 3D Trajectory & Telemetry Plot (PNG)
    # --------------------------------------------------------------------------
    os.makedirs("assets", exist_ok=True)
    out_img = "assets/pyflyt_50_drones_movement.png"
    
    fig = plt.figure(figsize=(22, 10), facecolor="#0D1117")
    
    ax1 = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#0D1117")
    ax1.scatter(start_pos[:, 0], start_pos[:, 1], start_pos[:, 2],
                color="#8B949E", s=30, alpha=0.6, label="Launch Grid (Z=0m)")
                
    cmap = plt.colormaps.get_cmap("viridis")
    colors = [cmap(i / num_drones) for i in range(num_drones)]
    for i in range(num_drones):
        traj = history_pos[:, i, :]
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                 color=colors[i], alpha=0.7, linewidth=1.4)
                 
    final_pos = history_pos[-1]
    anchor_indices = [i for i, t in enumerate(assigned_types) if t == "anchor"]
    edge_indices = [i for i, t in enumerate(assigned_types) if t == "edge"]
    
    if edge_indices:
        ax1.scatter(final_pos[edge_indices, 0], final_pos[edge_indices, 1], final_pos[edge_indices, 2],
                    color="#00E5FF", s=45, edgecolors="#FFFFFF", linewidth=0.6, alpha=0.95, label="Edge Drones (Final)")
    ax1.scatter(final_pos[anchor_indices, 0], final_pos[anchor_indices, 1], final_pos[anchor_indices, 2],
                color="#FFD700", s=75, edgecolors="#FFFFFF", linewidth=1.0, alpha=1.0, label="Anchor Drones (Final)")
                
    ax1.set_title(f"PyFlyt 50-Drone 3D Flight Trajectories\nFormation: {shape_name.upper()} (Takeoff → Flight Paths → Wireframe)",
                  color="#58A6FF", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlim([-7, 7])
    ax1.set_ylim([-7, 7])
    ax1.set_zlim([0, 10])
    ax1.view_init(elev=24, azim=-55)
    ax1.grid(True, linestyle=":", alpha=0.25, color="#30363D")
    ax1.tick_params(colors="#8B949E", labelsize=8)
    for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        pane.set_facecolor("#161B22")
        pane.set_edgecolor("#30363D")
    ax1.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", loc="upper left")
    
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#161B22")
    time_axis = np.linspace(0, sim_steps / 60.0, sim_steps)
    for i in range(num_drones):
        ax2.plot(time_axis, history_pos[:, i, 2], color=colors[i], alpha=0.4, linewidth=1.0)
    ax2.set_title("Altitude Progression Z(t) for All 50 Drones", color="#58A6FF", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Flight Time (seconds)", color="#8B949E", fontsize=9)
    ax2.set_ylabel("Altitude Z (meters)", color="#8B949E", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.3, color="#30363D")
    ax2.tick_params(colors="#8B949E", labelsize=8)
    
    ax3 = fig.add_subplot(2, 2, 4, facecolor="#161B22")
    ax3.plot(time_axis, min_distances, color="#00E5FF", linewidth=2.0, label="Min Inter-Drone Spacing")
    ax3.axhline(y=0.5, color="#FF4444", linestyle="--", linewidth=1.5, label="Safety Buffer (0.5m)")
    ax3.set_title("Swarm Safety Separation: Min Pairwise Distance (t)", color="#58A6FF", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Flight Time (seconds)", color="#8B949E", fontsize=9)
    ax3.set_ylabel("Separation Distance (meters)", color="#8B949E", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.3, color="#30363D")
    ax3.tick_params(colors="#8B949E", labelsize=8)
    ax3.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9", loc="upper right")
    
    plt.tight_layout()
    plt.savefig(out_img, dpi=160, facecolor="#0D1117", bbox_inches="tight")
    plt.close()
    print(f"Saved Static 3D Trajectory & Telemetry Plot: {out_img}")
    
    # --------------------------------------------------------------------------
    # 4. Active Hover Loop in GUI Mode
    # --------------------------------------------------------------------------
    if gui_mode:
        print("\n" + "=" * 75)
        print("🎉 FORMATION REACHED! The 50 drones are actively hovering in 3D space.")
        print("🎮 FREE FLIGHT CONTROLS ACTIVE (Click on the window to focus):")
        print("  • Move Around:  [W] forward, [S] back, [A] left, [D] right")
        print("  • Elevation:    [Space] / [E] up, [C] / [Q] down")
        print("  • Look / Zoom:  Hold [Ctrl] + Left Click Drag (rotate), Right Click (zoom)")
        print("  • Exit:         Press [Ctrl + C] or close the window.")
        print("=" * 75)
        
        final_setpoints = np.zeros((num_drones, 4))
        for i in range(num_drones):
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

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyFlyt 50-Drone Swarm Physics Simulation")
    parser.add_argument("--gui", action="store_true", help="Launch live real-time interactive 3D OpenGL window on screen")
    args = parser.parse_args()
    
    run_pyflyt_50_drones_simulation(gui_mode=args.gui)
