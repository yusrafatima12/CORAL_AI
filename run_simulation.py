import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

# 1. World load karo
model = mujoco.MjModel.from_xml_path("simulation/underwater_world.xml")
data = mujoco.MjData(model)
auv_id = model.body("auv").id
rock_ids = [model.geom(f"rock{i}").id for i in range(1, 6)]

# 2. Physics
mass = model.body_mass[auv_id]
buoyancy_force = mass * 9.81
drag_coefficient = 8.0

# 3. Jagahein
home = np.array([0.0, 0.0, 1.0])
goal = np.array([6.0, 4.5, 0.6])      # goal ab door hai

# 4. Controller
kp = 15.0
kd = 17.0
max_force = 15.0

# 5. Battery
battery_pct = 100.0
low_battery_limit = 30.0
returning = False

# 6. Sonar settings
sonar_range = 2.0        # 2 meter tak dekhta hai
influence = 1.6          # is faasle se obstacle dhakka dena shuru karta hai
rep_gain = 3.0           # dhakke ki taqat
swirl_gain = 8.0         # side se nikalne ki taqat
ray_angles = np.radians(np.arange(0, 360, 15))   # 24 rays, 360 degree

swirl_dir = 0.0          # 0 = abhi koi side nahi chuni
warned = False

# 7. Mission state
phase = 0                # 0 = goal ki taraf, 1 = home ki taraf
mission_done = False

# 8. Graph data
times, z_values, battery_values = [], [], []
xs, ys = [], []


def sonar_scan(pos):
    """24 rays bhejo, har ray ka faasla aur direction wapas do."""
    dists, dirs = [], []
    for a in ray_angles:
        d = np.array([np.cos(a), np.sin(a), 0.0])
        geomid = np.zeros(1, dtype=np.int32)
        r = mujoco.mj_ray(model, data, pos.copy(), d, None, 1, auv_id, geomid)
        dists.append(sonar_range if (r < 0 or r > sonar_range) else r)
        dirs.append(d)
    return np.array(dists), dirs


print("Mission start")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # 10 second ruko, camera set karo
    wait_start = time.time()
    while time.time() - wait_start < 10:
        viewer.sync()
        time.sleep(0.05)

    step = 0
    while viewer.is_running():
        position = data.qpos[:3].copy()
        velocity = data.qvel[:3].copy()

        # Abhi ka target
        if phase == 0 and not returning:
            target = goal
        else:
            target = home
        error = target - position
        to_target = error / (np.linalg.norm(error) + 1e-6)

        # --- SONAR ---
        dists, dirs = sonar_scan(position)

        repulsion = np.zeros(3)
        blocked = False
        for d, dist in zip(dirs, dists):
            if dist < influence:
                # Obstacle jitna qareeb, dhakka utna zyada
                repulsion += -d * rep_gain * (1.0 / max(dist, 0.15) - 1.0 / influence)
                if np.dot(d, to_target) > 0.7:
                    blocked = True   # obstacle seedhe raste mein hai

        if blocked:
            side = np.array([-to_target[1], to_target[0], 0.0])
            if swirl_dir == 0.0:
                # Wo side chuno jahan zyada khali jagah hai
                left = sum(x for d, x in zip(dirs, dists) if np.dot(d, side) > 0.5)
                right = sum(x for d, x in zip(dirs, dists) if np.dot(d, side) < -0.5)
                swirl_dir = 1.0 if left >= right else -1.0
            repulsion += swirl_gain * swirl_dir * side
            if not warned:
                print("SONAR: obstacle ahead, going around it...")
                warned = True
        else:
            swirl_dir = 0.0
            warned = False

        # --- CONTROLLER ---
        control = np.clip(kp * error - kd * velocity, -max_force, max_force)
        control[:2] = np.clip(control[:2], -6, 6)   # taake dhakka jeet sake
        force = control + repulsion
        force[2] = control[2]                        # upar neeche sirf controller
        force = np.clip(force, -max_force, max_force)

        # Forces lagao
        drag = -drag_coefficient * velocity
        data.xfrc_applied[auv_id, :] = 0
        data.xfrc_applied[auv_id, :3] = drag + force
        data.xfrc_applied[auv_id, 2] += buoyancy_force

        mujoco.mj_step(model, data)

        # Battery
        battery_pct -= 0.002 + 0.0005 * np.linalg.norm(force)
        if battery_pct < low_battery_limit and not returning:
            print("LOW BATTERY! Returning home...")
            returning = True

        # Graph data
        times.append(data.time)
        z_values.append(position[2])
        battery_values.append(battery_pct)
        xs.append(position[0])
        ys.append(position[1])

        # Goal par pohanche?
        if phase == 0 and not returning and np.linalg.norm(goal - position) < 0.2:
            print("Goal reached! Going back home...")
            phase = 1

        # Home par pohanche?
        if (phase == 1 or returning) and np.linalg.norm(home - position) < 0.2:
            if not mission_done:
                print("Mission complete: robot is back home")
                mission_done = True

        step += 1
        if step % 200 == 0:
            print("Pos:", position.round(2),
                  "| Battery:", round(battery_pct, 1), "%",
                  "| Nearest obstacle:", round(dists.min(), 2), "m")

        viewer.sync()

# 9. Graphs
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(times, z_values)
plt.xlabel("Time (s)")
plt.ylabel("Z position (m)")
plt.title("Depth vs Time")

plt.subplot(1, 3, 2)
plt.plot(times, battery_values, color="green")
plt.xlabel("Time (s)")
plt.ylabel("Battery (%)")
plt.title("Battery vs Time")

plt.subplot(1, 3, 3)
plt.plot(xs, ys, label="AUV path")
for gid in rock_ids:
    cx, cy = model.geom_pos[gid][:2]
    sx, sy = model.geom_size[gid][:2]
    plt.gca().add_patch(plt.Rectangle((cx - sx, cy - sy), 2 * sx, 2 * sy, color="red"))
plt.scatter([home[0]], [home[1]], c="green", s=80, label="Home")
plt.scatter([goal[0]], [goal[1]], c="gold", s=80, label="Goal")
plt.axis("equal")
plt.legend()
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Top view: path around obstacles")

plt.tight_layout()
plt.show()