import csv
import time
import numpy as np
import random

from dr_testbed.simulator import Simulator
from dr_testbed.render import Renderer
from dr_testbed.utility import load_waypoints, load_config_traf_lights, build_trajectory_from_waypoint
from dr_testbed.dynamic_model import KinematicBicycle 

# import your pilots
from dr_testbed.auto_pilot import PIDPilot, LQRPilot, RobustPilot, MPCPilot, DWAPilot

def run_evaluation(pilot_class, pilot_kwargs, scenario, max_steps=1800):
    """
    Runs one controller (pilot_class(**pilot_kwargs)) on the given scenario.
    Returns a dict of metrics.
    """
    # --- setup sim ---
    seed = 190
    way_points = load_waypoints(scenario["waypoints"])
    traf_lights = load_config_traf_lights(scenario["lights"])
    rng = random.Random(seed)
    start_wp = rng.choice(way_points)
    traj = build_trajectory_from_waypoint(start_wp, rng=rng)
    initial_heading = np.arctan2(start_wp.dir[1], start_wp.dir[0])
    vehicle_states = [(start_wp.pos, initial_heading)]
    sim = Simulator(num_vehicle=1, vehicle_states=vehicle_states, lights=traf_lights)
    renderer = Renderer(map_layout=scenario.get("map_layout", "map_1"))
    current_target_index = 0

    # instantiate pilot
    pilot = pilot_class(**pilot_kwargs)

    # metrics accumulators
    errors_ct = []      # cross-track error
    errors_hd = []      # heading error
    times = []          # compute time
    controls = []       # (v_cmd, steer)
    success = False

    for step in range(max_steps):
        # get current state
        ((x,y), psi) = vehicle_states[0]

        # proximity‐based waypoint progression
        target_pt = traj[current_target_index]
        dist = np.hypot(target_pt[0] - x, target_pt[1] - y)
        if dist < 0.1 and current_target_index < len(traj)-1:
            current_target_index += 1
            target_pt = traj[current_target_index]

        # build DummyWaypoint with pos & dir
        class DummyWP: pass
        wp = DummyWP()
        wp.pos = target_pt
        if current_target_index < len(traj)-1:
            nxt = traj[current_target_index+1]
            dx = nxt[0] - target_pt[0]
            dy = nxt[1] - target_pt[1]
            norm = np.hypot(dx, dy)
            wp.dir = (dx/norm, dy/norm) if norm>0 else (np.cos(psi), np.sin(psi))
        else:
            wp.dir = (np.cos(psi), np.sin(psi))

        # measure errors
        ct_err = np.hypot(x - wp.pos[0], y - wp.pos[1])
        psi_ref = np.arctan2(wp.pos[1]-y, wp.pos[0]-x)
        hd_err = ((psi - psi_ref + np.pi)%(2*np.pi)) - np.pi

        # controller timing
        t0 = time.perf_counter()
        v_cmd, steer_cmd, _ = pilot.step((x,y,psi), traj, wp)
        dt_ctrl = time.perf_counter() - t0

        # step sim
        vehicle_states, _ = sim.step([(v_cmd, steer_cmd)])
        renderer.render(vehicle_states, [], way_points, traj)

        # log metrics
        errors_ct.append(ct_err)
        errors_hd.append(abs(hd_err))
        times.append(dt_ctrl)
        controls.append((v_cmd, steer_cmd))

        # check success
        if step == max_steps -1 and ct_err < 0.15:
            success = True
            break

    # convert to arrays
    ct_arr = np.array(errors_ct)
    hd_arr = np.array(errors_hd)
    times_arr = np.array(times)
    steer_arr = np.array([c[1] for c in controls])

    # steering‐rate (deg) smoothness
    steer_diff = np.abs(np.diff(steer_arr))
    steer_rate_mean = np.degrees(steer_diff).mean() if steer_diff.size>0 else 0.0
    steer_rate_max  = np.degrees(steer_diff).max()  if steer_diff.size>0 else 0.0

    # tracking‐rate: % of steps where ct_err < 0.1 m
    track_rate = 100.0 * np.mean(ct_arr < 0.1) if ct_arr.size>0 else 0.0

    # summary metrics
    metrics = {
        "method":          pilot_class.__name__,
        "success":         success,
        "steps":           len(ct_arr),
        "ct_RMSE":         float(np.sqrt((ct_arr**2).mean())) if ct_arr.size>0 else None,
        "ct_max":          float(ct_arr.max())          if ct_arr.size>0 else None,
        "hd_RMSE":         float(np.sqrt((hd_arr**2).mean())) if hd_arr.size>0 else None,
        "hd_max":          float(hd_arr.max())          if hd_arr.size>0 else None,
        "track_rate_pct":  float(track_rate),   # % within 0.1 m
        "time_mean_ms":    float(times_arr.mean()*1e3) if times_arr.size>0 else None,
        "steer_mean_deg":  float(np.degrees(np.abs(steer_arr)).mean()) if steer_arr.size>0 else None,
        "steer_max_deg":   float(np.degrees(np.abs(steer_arr)).max()) if steer_arr.size>0 else None,
        "steer_rate_mean_deg": steer_rate_mean,
        "steer_rate_max_deg":  steer_rate_max,
        "v_mean":          float(np.mean([c[0] for c in controls])) if controls else None
    }
    return metrics

if __name__=="__main__":
    scenario = {
        "waypoints": "maps/square_intersection.json",
        "lights":    "maps/traffic_lights.json",
        "map_layout":"map_1"
    }

    pilots = [
        # Dynamic Window Approach
        (DWAPilot, dict(
            model=KinematicBicycle(wheel_base=0.15,
                                  max_steering_angel=np.radians(15),
                                  velocity_range=[-0.2, 0.5]),
            time_step=1/60,
            velocity=0.35,
            planning_horizon=30,
            steering_angle=15,
            num_samples=31,
            yaw_weight=0.3
        )),
        # PID Controller
        (PIDPilot, dict(
            kp_steer=1.5, ki_steer=0.05, kd_steer=0.1,
            kp_speed=1.2, ki_speed=0.1, kd_speed=0.01,
            desired_speed=0.35, time_step=1/60
        )),
        # LQR Controller
        (LQRPilot, dict(
            v0=0.35, L=0.15,
            Q=np.diag([10,1]), R=np.array([[1]]),
            dt=1/60, steer_limit_deg=15.0
        )),
        # Robust Controller
        (RobustPilot, dict(
            v0=0.35, L=0.15,
            lam=0.5, K=1.0, eta=0.2, phi=0.1,
            steer_limit_deg=15.0
        )),
        # MPC Controller
        (MPCPilot, dict(
            v0=0.35, L=0.15,
            dt=1/60, horizon=10,
            Q=np.diag([1,10]), R=np.array([[1]]),
            steer_limit_deg=15.0
        )),
    ]

    with open("performance.csv","w",newline="") as f:
        fieldnames = [
            "method","success","steps","ct_RMSE","ct_max",
            "hd_RMSE","hd_max","track_rate_pct","time_mean_ms",
            "steer_mean_deg","steer_max_deg",
            "steer_rate_mean_deg","steer_rate_max_deg","v_mean"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls, kw in pilots:
            m = run_evaluation(cls, kw, scenario)
            writer.writerow(m)
            print(f"{m['method']}: success={m['success']}, ct_RMSE={m['ct_RMSE']:.3f} m")
    print("All done. See performance.csv.")
